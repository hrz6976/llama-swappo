package proxy

import (
	"fmt"
	"math"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

const (
	gib = uint64(1024 * 1024 * 1024)
	kib = uint64(1024)
)

type gpuMemoryInfo struct {
	Index      int
	TotalBytes uint64
	FreeBytes  uint64
}

type smartAllocationDecision struct {
	Enabled          bool
	Reason           string
	Devices          []int
	ContextLength    int
	MaxRunning       int
	TPSize           int
	MemFraction      float64
	EstimatedWeights uint64
	EstimatedKV      uint64
}

var discoverGPUInfo = discoverNvidiaSMIGPUs

func applySmartAllocation(modelID string, modelConfig config.ModelConfig, args []string, env []string) ([]string, []string, smartAllocationDecision, error) {
	policy := modelConfig.SmartAlloc
	if !policy.Enabled {
		return args, env, smartAllocationDecision{Enabled: false, Reason: "disabled"}, nil
	}
	backend := strings.ToLower(strings.TrimSpace(policy.Backend))
	if backend != "" && backend != "sglang" {
		return args, env, smartAllocationDecision{Enabled: false, Reason: fmt.Sprintf("unsupported smartAlloc backend %q", policy.Backend)}, nil
	}

	gpus, err := discoverGPUInfo()
	if err != nil {
		return args, env, smartAllocationDecision{Enabled: false, Reason: fmt.Sprintf("gpu discovery failed: %v", err)}, nil
	}
	if len(gpus) == 0 {
		return args, env, smartAllocationDecision{Enabled: false, Reason: "no GPUs discovered"}, nil
	}

	decision, err := chooseSmartAllocation(modelID, modelConfig, args, gpus)
	if err != nil {
		return args, env, decision, err
	}

	args = setArg(args, []string{"--context-length", "--max-model-len"}, "--context-length", strconv.Itoa(decision.ContextLength))
	args = setArg(args, []string{"--max-running-requests"}, "--max-running-requests", strconv.Itoa(decision.MaxRunning))
	args = setArg(args, []string{"--tp-size", "--tp", "--tensor-parallel-size"}, "--tp-size", strconv.Itoa(decision.TPSize))
	args = setArg(args, []string{"--mem-fraction-static"}, "--mem-fraction-static", fmt.Sprintf("%.2f", decision.MemFraction))
	env = setEnv(env, "CUDA_VISIBLE_DEVICES", joinInts(decision.Devices))

	return args, env, decision, nil
}

func chooseSmartAllocation(modelID string, modelConfig config.ModelConfig, args []string, gpus []gpuMemoryInfo) (smartAllocationDecision, error) {
	policy := normalizeSmartAllocPolicy(modelConfig, args)
	if policy.ModelSizeBytes == 0 {
		return smartAllocationDecision{Enabled: true, Reason: "modelSizeBytes is required"}, fmt.Errorf("smartAlloc for %s requires modelSizeBytes", modelID)
	}

	sort.Slice(gpus, func(i, j int) bool {
		if gpus[i].FreeBytes == gpus[j].FreeBytes {
			return gpus[i].Index < gpus[j].Index
		}
		return gpus[i].FreeBytes > gpus[j].FreeBytes
	})

	maxGPUs := len(gpus)
	if policy.MaxGPUs > 0 && policy.MaxGPUs < maxGPUs {
		maxGPUs = policy.MaxGPUs
	}

	weights := uint64(math.Ceil(float64(policy.ModelSizeBytes) * 1.15))
	best := smartAllocationDecision{}

	if policy.PreferSpread {
		if decision, ok := fitOnGPUSet(gpus[:maxGPUs], policy, weights); ok {
			best = decision
		}
	} else {
		for n := 1; n <= maxGPUs; n++ {
			if decision, ok := fitOnGPUSet(gpus[:n], policy, weights); ok {
				if decision.ContextLength >= policy.PreferredContext {
					best = decision
					break
				}
				if len(best.Devices) == 0 || decision.ContextLength > best.ContextLength {
					best = decision
				}
			}
		}
	}

	if len(best.Devices) == 0 {
		decision, _ := fitOnGPUSet(gpus[:maxGPUs], policy, weights)
		if len(decision.Devices) == 0 {
			decision = minimumAllocation(gpus[:maxGPUs], policy, weights)
		}
		decision.Reason = "no exact fit found; using minimum viable allocation"
		best = decision
	}

	best.Enabled = true
	best.TPSize = len(best.Devices)
	best.EstimatedWeights = weights
	return best, nil
}

func normalizeSmartAllocPolicy(modelConfig config.ModelConfig, args []string) config.SmartAllocConfig {
	policy := modelConfig.SmartAlloc
	if policy.PreferredContext <= 0 {
		policy.PreferredContext = intArg(args, []string{"--context-length", "--max-model-len"}, 32768)
	}
	if policy.MaxContext > 0 && policy.PreferredContext > policy.MaxContext {
		policy.PreferredContext = policy.MaxContext
	}
	if policy.MinContext <= 0 {
		policy.MinContext = 4096
	}
	if policy.MinContext > policy.PreferredContext {
		policy.MinContext = policy.PreferredContext
	}
	if policy.MaxParallel <= 0 {
		policy.MaxParallel = intArg(args, []string{"--max-running-requests"}, 4)
	}
	if policy.MinParallel <= 0 {
		policy.MinParallel = 1
	}
	if policy.MinParallel > policy.MaxParallel {
		policy.MinParallel = policy.MaxParallel
	}
	if policy.ReserveBytes == 0 {
		policy.ReserveBytes = 6 * gib
	}
	if policy.OverheadBytes == 0 {
		policy.OverheadBytes = 1 * gib
	}
	if policy.KVBytesPerToken == 0 {
		policy.KVBytesPerToken = inferKVBytesPerToken(modelConfig)
	}
	return policy
}

func fitOnGPUSet(gpus []gpuMemoryInfo, policy config.SmartAllocConfig, weights uint64) (smartAllocationDecision, bool) {
	if len(gpus) == 0 {
		return smartAllocationDecision{}, false
	}

	var usable uint64
	minMemFraction := 0.92
	devices := make([]int, 0, len(gpus))
	for _, gpu := range gpus {
		devices = append(devices, gpu.Index)
		if gpu.FreeBytes > policy.ReserveBytes {
			usable += gpu.FreeBytes - policy.ReserveBytes
		}
		if gpu.TotalBytes > 0 {
			fraction := 1.0 - float64(policy.ReserveBytes)/float64(gpu.TotalBytes)
			if fraction < minMemFraction {
				minMemFraction = fraction
			}
		}
	}

	if usable <= weights+policy.OverheadBytes {
		return smartAllocationDecision{}, false
	}
	kvBudget := usable - weights - policy.OverheadBytes
	ctxCapacity := int(kvBudget / (policy.KVBytesPerToken * uint64(policy.MinParallel)))
	if ctxCapacity < policy.MinContext {
		return smartAllocationDecision{}, false
	}

	ctx := min(policy.PreferredContext, ctxCapacity)
	maxParallelByMemory := int(kvBudget / (policy.KVBytesPerToken * uint64(ctx)))
	running := min(policy.MaxParallel, max(policy.MinParallel, maxParallelByMemory))

	if minMemFraction < 0.60 {
		minMemFraction = 0.60
	}
	if minMemFraction > 0.95 {
		minMemFraction = 0.95
	}

	return smartAllocationDecision{
		Enabled:       true,
		Devices:       devices,
		ContextLength: ctx,
		MaxRunning:    running,
		TPSize:        len(devices),
		MemFraction:   minMemFraction,
		EstimatedKV:   uint64(ctx) * uint64(running) * policy.KVBytesPerToken,
	}, true
}

func minimumAllocation(gpus []gpuMemoryInfo, policy config.SmartAllocConfig, weights uint64) smartAllocationDecision {
	devices := make([]int, 0, len(gpus))
	minMemFraction := 0.85
	for _, gpu := range gpus {
		devices = append(devices, gpu.Index)
		if gpu.TotalBytes > 0 {
			fraction := 1.0 - float64(policy.ReserveBytes)/float64(gpu.TotalBytes)
			if fraction < minMemFraction {
				minMemFraction = fraction
			}
		}
	}
	if minMemFraction < 0.60 {
		minMemFraction = 0.60
	}
	return smartAllocationDecision{
		Enabled:          true,
		Devices:          devices,
		ContextLength:    policy.MinContext,
		MaxRunning:       policy.MinParallel,
		TPSize:           len(devices),
		MemFraction:      minMemFraction,
		EstimatedWeights: weights,
		EstimatedKV:      uint64(policy.MinContext) * uint64(policy.MinParallel) * policy.KVBytesPerToken,
	}
}

func discoverNvidiaSMIGPUs() ([]gpuMemoryInfo, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=index,memory.total,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, err
	}

	rows := strings.Split(strings.TrimSpace(string(out)), "\n")
	gpus := make([]gpuMemoryInfo, 0, len(rows))
	for _, row := range rows {
		parts := strings.Split(row, ",")
		if len(parts) != 3 {
			continue
		}
		index, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
		totalMiB, err2 := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
		freeMiB, err3 := strconv.ParseUint(strings.TrimSpace(parts[2]), 10, 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		gpus = append(gpus, gpuMemoryInfo{
			Index:      index,
			TotalBytes: totalMiB * 1024 * 1024,
			FreeBytes:  freeMiB * 1024 * 1024,
		})
	}
	return gpus, nil
}

func inferKVBytesPerToken(modelConfig config.ModelConfig) uint64 {
	paramSize, _ := modelConfig.Metadata["parameterSize"].(string)
	billions := parseParameterBillions(paramSize)
	switch {
	case billions > 70:
		return 1024 * kib
	case billions > 34:
		return 768 * kib
	case billions > 14:
		return 512 * kib
	case billions > 8:
		return 256 * kib
	default:
		return 128 * kib
	}
}

func parseParameterBillions(value string) float64 {
	value = strings.TrimSpace(strings.ToUpper(value))
	value = strings.TrimSuffix(value, "B")
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0
	}
	return parsed
}

func setArg(args []string, aliases []string, canonical string, value string) []string {
	out := make([]string, 0, len(args)+2)
	skipNext := false
	for i := 0; i < len(args); i++ {
		if skipNext {
			skipNext = false
			continue
		}
		arg := args[i]
		matched := false
		for _, alias := range aliases {
			if arg == alias {
				if i+1 < len(args) {
					skipNext = true
				}
				matched = true
				break
			}
			if strings.HasPrefix(arg, alias+"=") {
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, arg)
		}
	}
	return append(out, canonical, value)
}

func intArg(args []string, aliases []string, fallback int) int {
	for i := 0; i < len(args); i++ {
		for _, alias := range aliases {
			if args[i] == alias && i+1 < len(args) {
				if v, err := strconv.Atoi(args[i+1]); err == nil {
					return v
				}
			}
			if strings.HasPrefix(args[i], alias+"=") {
				if v, err := strconv.Atoi(strings.TrimPrefix(args[i], alias+"=")); err == nil {
					return v
				}
			}
		}
	}
	return fallback
}

func setEnv(env []string, key string, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		if !strings.HasPrefix(item, prefix) {
			out = append(out, item)
		}
	}
	return append(out, prefix+value)
}

func joinInts(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}
