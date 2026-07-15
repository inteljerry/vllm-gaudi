from vllm.model_executor.models.registry import ModelRegistry


def _register_model_stock():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")

    from vllm_gaudi.models.qwen2_5_vl import HpuQwen2_5_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen2_5_vl:HpuQwen2_5_VLForConditionalGeneration")

    from vllm_gaudi.models.ernie45_vl import HpuErnie4_5_VLMoeForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Ernie4_5_VLMoeForConditionalGeneration",
                                 "vllm_gaudi.models.ernie45_vl:HpuErnie4_5_VLMoeForConditionalGeneration")

    from vllm_gaudi.models.ovis import HpuOvis  # noqa: F401
    ModelRegistry.register_model("Ovis", "vllm_gaudi.models.ovis:HpuOvis")

    from vllm_gaudi.models.qwen3_vl_moe import HpuQwen3_VLMoeForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen3VLMoeForConditionalGeneration",
                                 "vllm_gaudi.models.qwen3_vl_moe:HpuQwen3_VLMoeForConditionalGeneration")

    from vllm_gaudi.models.hunyuan_v1 import HpuHunYuanDenseV1ForCausalLM  # noqa: F401
    ModelRegistry.register_model("HunYuanDenseV1ForCausalLM",
                                 "vllm_gaudi.models.hunyuan_v1:HpuHunYuanDenseV1ForCausalLM")

    from vllm_gaudi.models.hunyuan_v1 import HpuHunYuanMoEV1ForCausalLM  # noqa: F401
    ModelRegistry.register_model("HunYuanMoEV1ForCausalLM", "vllm_gaudi.models.hunyuan_v1:HpuHunYuanMoEV1ForCausalLM")

    from vllm_gaudi.models.minimax_m2 import HpuMiniMaxM2ForCausalLM  # noqa: F401
    ModelRegistry.register_model("MiniMaxM2ForCausalLM", "vllm_gaudi.models.minimax_m2:HpuMiniMaxM2ForCausalLM")
    from vllm_gaudi.models.pixtral import HPUPixtralForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("PixtralForConditionalGeneration",
                                 "vllm_gaudi.models.pixtral:HPUPixtralForConditionalGeneration")

    from vllm_gaudi.models.dots_ocr import HpuDotsOCRForCausalLM  # noqa: F401
    ModelRegistry.register_model("DotsOCRForCausalLM", "vllm_gaudi.models.dots_ocr:HpuDotsOCRForCausalLM")

    from vllm_gaudi.models.seed_oss import HpuSeedOssForCausalLM  # noqa: F401
    ModelRegistry.register_model("SeedOssForCausalLM", "vllm_gaudi.models.seed_oss:HpuSeedOssForCausalLM")

    from vllm_gaudi.models.qwen3_moe import HpuQwen3MoeForCausalLM  # noqa: F401
    ModelRegistry.register_model("Qwen3MoeForCausalLM", "vllm_gaudi.models.qwen3_moe:HpuQwen3MoeForCausalLM")

    from vllm_gaudi.models.llama4 import HpuLlama4ForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Llama4ForConditionalGeneration",
                                 "vllm_gaudi.models.llama4:HpuLlama4ForConditionalGeneration")

    import vllm_gaudi.models.deepseek_v2  # noqa: F401

    from vllm_gaudi.models.deepseek_ocr import HpuDeepseekOCRForCausalLM  # noqa: F401
    ModelRegistry.register_model("DeepseekOCRForCausalLM", "vllm_gaudi.models.deepseek_ocr:HpuDeepseekOCRForCausalLM")

    import vllm_gaudi.models.gptoss_mxfp4  # noqa: F401
    import vllm_gaudi.models.qwen3_next  # noqa: F401
    import vllm_gaudi.models.qwen3_5  # noqa: F401


def register_model():
    # MiniMax-M3, registered by LAZY STRING (module imported only when a
    # request actually uses the arch), so registration never eagerly pulls
    # torchvision.
    #   * MiniMaxM3SparseForConditionalGeneration (the checkpoint's own
    #     arch) -> the VL model (vision tower + text backbone). Its native,
    #     torchvision-free image processor makes image requests work; a
    #     text-only request through it simply skips the vision path.
    #   * MiniMaxM3SparseForCausalLM -> the lean text-only class (use via
    #     --hf-overrides architectures for a text-only serve with no vision
    #     tower in memory, e.g. the max-KV long-context profile).
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_gaudi.models.minimax_m3_vl:MiniMaxM3SparseForConditionalGeneration")
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_gaudi.models.minimax_m3:HpuMiniMaxM3SparseForCausalLM")
    # Override the stock eagle3 llama draft with the fc_norm-capable
    # variant (newer draft checkpoints, e.g. Inferact/MiniMax-M3-EAGLE3,
    # carry per-aux fc_norm.{0,1,2} weights the pinned class cannot load).
    ModelRegistry.register_model(
        "LlamaForCausalLMEagle3",
        "vllm_gaudi.models.llama_eagle3_fcnorm:Eagle3LlamaForCausalLMFcNorm")
    try:
        _register_model_stock()
    except Exception as e:  # optional (e.g. VL) models may need torchvision
        import logging
        logging.getLogger(__name__).warning("some plugin models skipped: %s", e)
