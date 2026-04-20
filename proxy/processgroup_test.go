package proxy

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var processGroupTestConfig = config.AddDefaultGroupToConfig(config.Config{
	HealthCheckTimeout: 15,
	Models: map[string]config.ModelConfig{
		"model1": getTestSimpleResponderConfig("model1"),
		"model2": getTestSimpleResponderConfig("model2"),
		"model3": getTestSimpleResponderConfig("model3"),
		"model4": getTestSimpleResponderConfig("model4"),
		"model5": getTestSimpleResponderConfig("model5"),
	},
	Groups: map[string]config.GroupConfig{
		"G1": {
			Swap:      true,
			Exclusive: true,
			Members:   []string{"model1", "model2"},
		},
		"G2": {
			Swap:      false,
			Exclusive: true,
			Members:   []string{"model3", "model4"},
		},
	},
})

func TestProcessGroup_DefaultHasCorrectModel(t *testing.T) {
	pg := NewProcessGroup(config.DEFAULT_GROUP_ID, processGroupTestConfig, testLogger, testLogger)
	assert.True(t, pg.HasMember("model5"))
}

func TestProcessGroup_HasMember(t *testing.T) {
	pg := NewProcessGroup("G1", processGroupTestConfig, testLogger, testLogger)
	assert.True(t, pg.HasMember("model1"))
	assert.True(t, pg.HasMember("model2"))
	assert.False(t, pg.HasMember("model3"))
}

// TestProcessGroup_ProxyRequestSwapIsTrueParallel tests that when swap is true
// and multiple requests are made in parallel, only one process is running at a time.
func TestProcessGroup_ProxyRequestSwapIsTrueParallel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow test")
	}

	var processGroupTestConfig = config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			// use the same listening so if a model is already running, it will fail
			// this is a way to test that swap isolation is working
			// properly when there are parallel requests made at the
			// same time.
			"model1": getTestSimpleResponderConfigPort("model1", 9832),
			"model2": getTestSimpleResponderConfigPort("model2", 9832),
			"model3": getTestSimpleResponderConfigPort("model3", 9832),
			"model4": getTestSimpleResponderConfigPort("model4", 9832),
			"model5": getTestSimpleResponderConfigPort("model5", 9832),
		},
		Groups: map[string]config.GroupConfig{
			"G1": {
				Swap:    true,
				Members: []string{"model1", "model2", "model3", "model4", "model5"},
			},
		},
	})

	pg := NewProcessGroup("G1", processGroupTestConfig, testLogger, testLogger)
	defer pg.StopProcesses(StopWaitForInflightRequest)

	tests := []string{"model1", "model2", "model3", "model4", "model5"}

	var wg sync.WaitGroup

	wg.Add(len(tests))
	for _, modelName := range tests {
		go func(modelName string) {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			assert.NoError(t, pg.ProxyRequest(modelName, w, req))
			assert.Equal(t, http.StatusOK, w.Code)
			assert.Contains(t, w.Body.String(), modelName)
		}(modelName)
	}
	wg.Wait()
}

func TestProcessGroup_ProxyRequestSwapIsFalse(t *testing.T) {
	pg := NewProcessGroup("G2", processGroupTestConfig, testLogger, testLogger)
	defer pg.StopProcesses(StopWaitForInflightRequest)

	tests := []string{"model3", "model4"}

	for _, modelName := range tests {
		t.Run(modelName, func(t *testing.T) {
			reqBody := `{"x", "y"}`
			req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(reqBody))
			w := httptest.NewRecorder()
			assert.NoError(t, pg.ProxyRequest(modelName, w, req))
			assert.Equal(t, http.StatusOK, w.Code)
			assert.Contains(t, w.Body.String(), modelName)
		})
	}

	// make sure all the processes are running
	for _, process := range pg.processes {
		assert.Equal(t, StateReady, process.CurrentState())
	}
}

func TestProcessGroup_AutoScaleUsesBaseReplicaOnColdStart(t *testing.T) {
	modelConfig := getTestSimpleResponderConfig("autoscale-cold")
	modelConfig.AutoScale = config.AutoScaleConfig{
		Enabled:           true,
		MaxReplicas:       2,
		CooldownSeconds:   1,
		ScaleUpQueueRatio: 0.5,
	}
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"autoscale-cold": modelConfig,
		},
		Groups: map[string]config.GroupConfig{
			"G": {
				Swap:    false,
				Members: []string{"autoscale-cold"},
			},
		},
	})

	pg := NewProcessGroup("G", cfg, testLogger, testLogger)
	defer pg.StopProcesses(StopWaitForInflightRequest)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	require.NoError(t, pg.ProxyRequest("autoscale-cold", w, req))
	assert.Equal(t, http.StatusOK, w.Code)
	require.Len(t, pg.replicaSets["autoscale-cold"].Replicas, 1)
	assert.Equal(t, "autoscale-cold", pg.replicaSets["autoscale-cold"].Replicas[0].ID)
}

func TestProcessGroup_AutoScaleStartsReplicaWhenSaturated(t *testing.T) {
	modelConfig := getTestSimpleResponderConfig("autoscale-hot")
	modelConfig.ConcurrencyLimit = 1
	modelConfig.AutoScale = config.AutoScaleConfig{
		Enabled:           true,
		MaxReplicas:       2,
		CooldownSeconds:   1,
		ScaleUpQueueRatio: 0.5,
	}
	cfg := config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"autoscale-hot": modelConfig,
		},
		Groups: map[string]config.GroupConfig{
			"G": {
				Swap:    false,
				Members: []string{"autoscale-hot"},
			},
		},
	})

	pg := NewProcessGroup("G", cfg, testLogger, testLogger)
	defer pg.StopProcesses(StopWaitForInflightRequest)

	firstDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		req := httptest.NewRequest("GET", "/slow-respond?echo=first&delay=700ms", nil)
		w := httptest.NewRecorder()
		_ = pg.ProxyRequest("autoscale-hot", w, req)
		firstDone <- w
	}()

	require.Eventually(t, func() bool {
		return pg.replicaSets["autoscale-hot"].Replicas[0].InFlightRequests() == 1
	}, 2*time.Second, 10*time.Millisecond)

	secondReq := httptest.NewRequest("GET", "/slow-respond?echo=second&delay=10ms", nil)
	second := httptest.NewRecorder()
	require.NoError(t, pg.ProxyRequest("autoscale-hot", second, secondReq))
	assert.Equal(t, http.StatusOK, second.Code)
	assert.Contains(t, second.Body.String(), "second")

	require.Len(t, pg.replicaSets["autoscale-hot"].Replicas, 2)
	assert.Equal(t, "autoscale-hot#1", pg.replicaSets["autoscale-hot"].Replicas[1].ID)
	assert.NotEqual(t, pg.replicaSets["autoscale-hot"].Replicas[0].config.Proxy, pg.replicaSets["autoscale-hot"].Replicas[1].config.Proxy)

	select {
	case first := <-firstDone:
		assert.Equal(t, http.StatusOK, first.Code)
		assert.Contains(t, first.Body.String(), "first")
	case <-time.After(10 * time.Second):
		t.Fatal("first request did not complete")
	}
}
