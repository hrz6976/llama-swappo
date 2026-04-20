package proxy

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

type ReplicaSet struct {
	ModelID       string
	Replicas      []*Process
	NextReplicaID int
	LastScaleUp   time.Time
	LastScaleDown time.Time
}

type ProcessGroup struct {
	sync.Mutex

	config     config.Config
	id         string
	swap       bool
	exclusive  bool
	persistent bool

	proxyLogger    *LogMonitor
	upstreamLogger *LogMonitor

	// map of current processes
	processes       map[string]*Process
	replicaSets     map[string]*ReplicaSet
	lastUsedProcess string
}

func NewProcessGroup(id string, config config.Config, proxyLogger *LogMonitor, upstreamLogger *LogMonitor) *ProcessGroup {
	groupConfig, ok := config.Groups[id]
	if !ok {
		panic("Unable to find configuration for group id: " + id)
	}

	pg := &ProcessGroup{
		id:             id,
		config:         config,
		swap:           groupConfig.Swap,
		exclusive:      groupConfig.Exclusive,
		persistent:     groupConfig.Persistent,
		proxyLogger:    proxyLogger,
		upstreamLogger: upstreamLogger,
		processes:      make(map[string]*Process),
		replicaSets:    make(map[string]*ReplicaSet),
	}

	// Create a Process for each member in the group
	for _, modelID := range groupConfig.Members {
		modelConfig, modelID, _ := pg.config.FindConfig(modelID)
		processLogger := NewLogMonitorWriter(upstreamLogger)
		process := NewProcess(modelID, pg.config.HealthCheckTimeout, modelConfig, processLogger, pg.proxyLogger)
		pg.processes[modelID] = process
		pg.replicaSets[modelID] = &ReplicaSet{
			ModelID:       modelID,
			Replicas:      []*Process{process},
			NextReplicaID: 1,
		}
	}

	return pg
}

// ProxyRequest proxies a request to the specified model
func (pg *ProcessGroup) ProxyRequest(modelID string, writer http.ResponseWriter, request *http.Request) error {
	if !pg.HasMember(modelID) {
		return fmt.Errorf("model %s not part of group %s", modelID, pg.id)
	}

	if pg.swap {
		pg.Lock()
		if pg.lastUsedProcess != modelID {

			// is there something already running?
			if pg.lastUsedProcess != "" {
				pg.stopReplicasLocked(pg.lastUsedProcess, StopWaitForInflightRequest)
			}

			// wait for the request to the new model to be fully handled
			// and prevent race conditions see issue #277
			process := pg.selectReplicaLocked(modelID)
			process.ProxyRequest(writer, request)
			pg.lastUsedProcess = modelID

			// short circuit and exit
			pg.Unlock()
			return nil
		}
		pg.Unlock()
	}

	process := pg.selectReplica(modelID)
	process.ProxyRequest(writer, request)
	return nil
}

func (pg *ProcessGroup) selectReplica(modelID string) *Process {
	pg.Lock()
	defer pg.Unlock()
	return pg.selectReplicaLocked(modelID)
}

func (pg *ProcessGroup) selectReplicaLocked(modelID string) *Process {
	rs := pg.replicaSets[modelID]
	if rs == nil || len(rs.Replicas) == 0 {
		return pg.processes[modelID]
	}

	policy := normalizedAutoScaleConfig(pg.config.Models[modelID].AutoScale)
	best := leastLoadedReplica(rs.Replicas)
	if !policy.Enabled {
		return best
	}

	pg.scaleDownIdleReplicasLocked(rs, policy)
	if best.CurrentState() == StateStopped && best != pg.processes[modelID] {
		return best
	}
	if pg.shouldScaleUpLocked(rs, policy) {
		if replica, err := pg.createReplicaLocked(rs); err == nil {
			pg.proxyLogger.Infof("<%s> autoscale started replica %s", modelID, replica.ID)
			rs.LastScaleUp = time.Now()
			return replica
		} else {
			pg.proxyLogger.Warnf("<%s> autoscale scale-up skipped: %v", modelID, err)
		}
	}

	return leastLoadedReplica(rs.Replicas)
}

func normalizedAutoScaleConfig(policy config.AutoScaleConfig) config.AutoScaleConfig {
	if policy.MinReplicas <= 0 {
		policy.MinReplicas = 1
	}
	if policy.MaxReplicas <= 0 {
		policy.MaxReplicas = policy.MinReplicas
	}
	if policy.MaxReplicas < policy.MinReplicas {
		policy.MaxReplicas = policy.MinReplicas
	}
	if policy.ScaleUpQueueRatio <= 0 {
		policy.ScaleUpQueueRatio = 0.75
	}
	if policy.ScaleUpQueueRatio > 1 {
		policy.ScaleUpQueueRatio = 1
	}
	if policy.ScaleDownIdleSeconds <= 0 {
		policy.ScaleDownIdleSeconds = 180
	}
	if policy.CooldownSeconds <= 0 {
		policy.CooldownSeconds = 30
	}
	return policy
}

func leastLoadedReplica(replicas []*Process) *Process {
	var best *Process
	bestScore := float64(1 << 30)
	for _, replica := range replicas {
		state := replica.CurrentState()
		if state == StateShutdown || state == StateStopping {
			continue
		}
		score := replica.QueueRatio()
		if state == StateStarting {
			score += 0.5
		}
		if !replica.CanAcceptRequest() {
			score += 1
		}
		if best == nil || score < bestScore {
			best = replica
			bestScore = score
		}
	}
	if best != nil {
		return best
	}
	return replicas[0]
}

func (pg *ProcessGroup) shouldScaleUpLocked(rs *ReplicaSet, policy config.AutoScaleConfig) bool {
	if len(rs.Replicas) >= policy.MaxReplicas {
		return false
	}
	if !rs.LastScaleUp.IsZero() && time.Since(rs.LastScaleUp) < time.Duration(policy.CooldownSeconds)*time.Second {
		return false
	}

	active := 0
	for _, replica := range rs.Replicas {
		state := replica.CurrentState()
		if state != StateReady && state != StateStarting {
			continue
		}
		active++
		if replica.QueueRatio() < policy.ScaleUpQueueRatio && replica.CanAcceptRequest() {
			return false
		}
	}
	return active > 0
}

func (pg *ProcessGroup) scaleDownIdleReplicasLocked(rs *ReplicaSet, policy config.AutoScaleConfig) {
	if len(rs.Replicas) <= policy.MinReplicas {
		return
	}
	if !rs.LastScaleDown.IsZero() && time.Since(rs.LastScaleDown) < time.Duration(policy.CooldownSeconds)*time.Second {
		return
	}

	idleFor := time.Duration(policy.ScaleDownIdleSeconds) * time.Second
	removed := 0
	removable := len(rs.Replicas) - policy.MinReplicas
	keep := rs.Replicas[:0]
	for _, replica := range rs.Replicas {
		if removed >= removable {
			keep = append(keep, replica)
			continue
		}
		if replica == pg.processes[rs.ModelID] {
			keep = append(keep, replica)
			continue
		}
		if replica.CurrentState() == StateReady &&
			replica.InFlightRequests() == 0 &&
			time.Since(replica.getLastRequestHandled()) > idleFor {
			pg.proxyLogger.Infof("<%s> autoscale stopping idle replica %s", rs.ModelID, replica.ID)
			go replica.Stop()
			rs.LastScaleDown = time.Now()
			removed++
			continue
		}
		keep = append(keep, replica)
	}
	rs.Replicas = keep
}

func (pg *ProcessGroup) createReplicaLocked(rs *ReplicaSet) (*Process, error) {
	baseConfig, ok := pg.config.Models[rs.ModelID]
	if !ok {
		return nil, fmt.Errorf("model config not found")
	}

	replicaIndex := rs.NextReplicaID
	replicaConfig, err := modelConfigForReplica(baseConfig)
	if err != nil {
		return nil, err
	}

	replicaID := fmt.Sprintf("%s#%d", rs.ModelID, replicaIndex)
	processLogger := NewLogMonitorWriter(pg.upstreamLogger)
	replica := NewProcess(replicaID, pg.config.HealthCheckTimeout, replicaConfig, processLogger, pg.proxyLogger)
	rs.Replicas = append(rs.Replicas, replica)
	rs.NextReplicaID++
	return replica, nil
}

func modelConfigForReplica(base config.ModelConfig) (config.ModelConfig, error) {
	replica := base
	proxyURL, err := url.Parse(base.Proxy)
	if err != nil {
		return replica, fmt.Errorf("invalid proxy URL %q: %w", base.Proxy, err)
	}
	oldPort := proxyURL.Port()
	if oldPort == "" {
		return replica, fmt.Errorf("proxy URL %q has no port to rewrite for replica", base.Proxy)
	}

	newPort, err := allocateReplicaPort()
	if err != nil {
		return replica, err
	}
	newPortStr := strconv.Itoa(newPort)
	replica.Cmd = strings.ReplaceAll(replica.Cmd, oldPort, newPortStr)
	replica.CmdStop = strings.ReplaceAll(replica.CmdStop, oldPort, newPortStr)
	replica.Proxy = strings.ReplaceAll(replica.Proxy, oldPort, newPortStr)
	replica.CheckEndpoint = strings.ReplaceAll(replica.CheckEndpoint, oldPort, newPortStr)
	for i, env := range replica.Env {
		replica.Env[i] = strings.ReplaceAll(env, oldPort, newPortStr)
	}
	return replica, nil
}

func allocateReplicaPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("allocate replica port: %w", err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func (pg *ProcessGroup) HasMember(modelName string) bool {
	return slices.Contains(pg.config.Groups[pg.id].Members, modelName)
}

func (pg *ProcessGroup) GetMember(modelName string) (*Process, bool) {
	if pg.HasMember(modelName) {
		pg.Lock()
		defer pg.Unlock()
		rs := pg.replicaSets[modelName]
		if rs != nil && len(rs.Replicas) > 0 {
			return leastLoadedReplica(rs.Replicas), true
		}
		return pg.processes[modelName], true
	}
	return nil, false
}

func (pg *ProcessGroup) StopProcess(modelID string, strategy StopStrategy) error {
	pg.Lock()

	if _, exists := pg.processes[modelID]; !exists {
		pg.Unlock()
		return fmt.Errorf("process not found for %s", modelID)
	}

	if pg.lastUsedProcess == modelID {
		pg.lastUsedProcess = ""
	}
	pg.Unlock()

	pg.stopReplicas(modelID, strategy)
	return nil
}

func (pg *ProcessGroup) StopProcesses(strategy StopStrategy) {
	pg.Lock()
	defer pg.Unlock()

	if len(pg.processes) == 0 {
		return
	}

	// stop Processes in parallel
	var wg sync.WaitGroup
	for _, rs := range pg.replicaSets {
		for _, process := range rs.Replicas {
			wg.Add(1)
			go func(process *Process) {
				defer wg.Done()
				stopProcess(process, strategy)
			}(process)
		}
	}
	wg.Wait()
}

func (pg *ProcessGroup) Shutdown() {
	var wg sync.WaitGroup
	for _, rs := range pg.replicaSets {
		for _, process := range rs.Replicas {
			wg.Add(1)
			go func(process *Process) {
				defer wg.Done()
				process.Shutdown()
			}(process)
		}
	}
	wg.Wait()
}

func (pg *ProcessGroup) stopReplicas(modelID string, strategy StopStrategy) {
	pg.Lock()
	rs := pg.replicaSets[modelID]
	if rs == nil {
		pg.Unlock()
		return
	}
	replicas := append([]*Process(nil), rs.Replicas...)
	pg.Unlock()

	var wg sync.WaitGroup
	for _, process := range replicas {
		wg.Add(1)
		go func(process *Process) {
			defer wg.Done()
			stopProcess(process, strategy)
		}(process)
	}
	wg.Wait()
}

func (pg *ProcessGroup) stopReplicasLocked(modelID string, strategy StopStrategy) {
	rs := pg.replicaSets[modelID]
	if rs == nil {
		return
	}
	var wg sync.WaitGroup
	for _, process := range rs.Replicas {
		wg.Add(1)
		go func(process *Process) {
			defer wg.Done()
			stopProcess(process, strategy)
		}(process)
	}
	wg.Wait()
}

func stopProcess(process *Process, strategy StopStrategy) {
	switch strategy {
	case StopImmediately:
		process.StopImmediately()
	default:
		process.Stop()
	}
}
