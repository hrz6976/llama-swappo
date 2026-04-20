package proxy

import (
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChooseSmartAllocationPrefersSmallestFittingGPUSet(t *testing.T) {
	modelConfig := config.ModelConfig{
		SmartAlloc: config.SmartAllocConfig{
			Enabled:          true,
			ModelSizeBytes:   5 * gib,
			PreferredContext: 131072,
			MinContext:       32768,
			MaxParallel:      4,
			MinParallel:      1,
			ReserveBytes:     6 * gib,
			OverheadBytes:    1 * gib,
			KVBytesPerToken:  128 * kib,
		},
		Metadata: map[string]any{"parameterSize": "7B"},
	}

	decision, err := chooseSmartAllocation("qwen2.5-7b", modelConfig, []string{"python3", "-m", "sglang.launch_server"}, []gpuMemoryInfo{
		{Index: 0, TotalBytes: 48 * gib, FreeBytes: 22 * gib},
		{Index: 1, TotalBytes: 48 * gib, FreeBytes: 44 * gib},
		{Index: 2, TotalBytes: 48 * gib, FreeBytes: 40 * gib},
	})
	require.NoError(t, err)

	assert.Equal(t, []int{1}, decision.Devices)
	assert.Equal(t, 1, decision.TPSize)
	assert.Equal(t, 131072, decision.ContextLength)
	assert.GreaterOrEqual(t, decision.MaxRunning, 1)
	assert.LessOrEqual(t, decision.MaxRunning, 4)
}

func TestChooseSmartAllocationAddsGPUsWhenContextDoesNotFit(t *testing.T) {
	modelConfig := config.ModelConfig{
		SmartAlloc: config.SmartAllocConfig{
			Enabled:          true,
			ModelSizeBytes:   20 * gib,
			PreferredContext: 131072,
			MinContext:       32768,
			MaxParallel:      4,
			MinParallel:      1,
			ReserveBytes:     6 * gib,
			OverheadBytes:    1 * gib,
			KVBytesPerToken:  512 * kib,
		},
		Metadata: map[string]any{"parameterSize": "34B"},
	}

	decision, err := chooseSmartAllocation("codellama-34b", modelConfig, nil, []gpuMemoryInfo{
		{Index: 0, TotalBytes: 48 * gib, FreeBytes: 37 * gib},
		{Index: 1, TotalBytes: 48 * gib, FreeBytes: 37 * gib},
		{Index: 2, TotalBytes: 48 * gib, FreeBytes: 37 * gib},
	})
	require.NoError(t, err)

	assert.Len(t, decision.Devices, 3)
	assert.Equal(t, 3, decision.TPSize)
	assert.Equal(t, 131072, decision.ContextLength)
}

func TestApplySmartAllocationRewritesSGLangArgsAndEnv(t *testing.T) {
	oldDiscover := discoverGPUInfo
	defer func() { discoverGPUInfo = oldDiscover }()
	discoverGPUInfo = func() ([]gpuMemoryInfo, error) {
		return []gpuMemoryInfo{{Index: 2, TotalBytes: 48 * gib, FreeBytes: 44 * gib}}, nil
	}

	args := []string{
		"python3", "-m", "sglang.launch_server",
		"--context-length", "32768",
		"--max-running-requests", "4",
		"--tp-size", "1",
		"--mem-fraction-static", "0.85",
	}
	env := []string{"CUDA_VISIBLE_DEVICES=0", "OTHER=1"}
	modelConfig := config.ModelConfig{
		SmartAlloc: config.SmartAllocConfig{
			Enabled:          true,
			Backend:          "sglang",
			ModelSizeBytes:   5 * gib,
			PreferredContext: 131072,
			MinContext:       32768,
			MaxParallel:      4,
			MinParallel:      1,
			ReserveBytes:     6 * gib,
			OverheadBytes:    1 * gib,
			KVBytesPerToken:  128 * kib,
		},
		Metadata: map[string]any{"parameterSize": "7B"},
	}

	gotArgs, gotEnv, decision, err := applySmartAllocation("qwen2.5-7b", modelConfig, args, env)
	require.NoError(t, err)

	assert.Equal(t, []int{2}, decision.Devices)
	assert.Contains(t, gotEnv, "CUDA_VISIBLE_DEVICES=2")
	assert.Contains(t, gotEnv, "OTHER=1")
	assert.Equal(t, 1, countArg(gotArgs, "--context-length"))
	assert.Contains(t, gotArgs, "131072")
	assert.Equal(t, 1, countArg(gotArgs, "--max-running-requests"))
	assert.Equal(t, 1, countArg(gotArgs, "--tp-size"))
	assert.Equal(t, 1, countArg(gotArgs, "--mem-fraction-static"))
}

func TestApplySmartAllocationSkipsUnsupportedBackend(t *testing.T) {
	args := []string{"custom-server", "--port", "1234"}
	env := []string{"CUDA_VISIBLE_DEVICES=0"}
	modelConfig := config.ModelConfig{
		SmartAlloc: config.SmartAllocConfig{
			Enabled: true,
			Backend: "custom",
		},
	}

	gotArgs, gotEnv, decision, err := applySmartAllocation("custom", modelConfig, args, env)
	require.NoError(t, err)

	assert.Equal(t, args, gotArgs)
	assert.Equal(t, env, gotEnv)
	assert.False(t, decision.Enabled)
	assert.Contains(t, decision.Reason, "unsupported")
}

func countArg(args []string, arg string) int {
	count := 0
	for _, item := range args {
		if item == arg || strings.HasPrefix(item, arg+"=") {
			count++
		}
	}
	return count
}
