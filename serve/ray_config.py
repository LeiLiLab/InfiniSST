#!/usr/bin/env python3
"""
Configuration Management for Ray-based InfiniSST Serving System
Ray版本InfiniSST服务系统的配置管理
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class ModelConfig:
    """单个模型的配置"""
    model_name: str
    model_path: str
    lora_path: Optional[str] = None
    state_dict_path: Optional[str] = None
    
    # Model-specific parameters
    latency_multiplier: int = 2
    max_new_tokens: int = 20
    beam_size: int = 4
    
    # Audio processing
    w2v2_path: str = "/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt"
    w2v2_type: str = "w2v2"
    ctc_finetuned: bool = True
    
    # Model architecture
    length_shrink_cfg: str = "[(1024,2,2)] * 2"
    block_size: int = 48
    max_cache_size: int = 576
    model_type: str = "w2v2_qwen25"
    rope: int = 1
    audio_normalize: int = 0
    
    # LLM parameters
    max_llm_cache_size: int = 1000
    always_cache_system_prompt: bool = True
    max_len_a: int = 10
    max_len_b: int = 20
    no_repeat_ngram_lookback: int = 100
    no_repeat_ngram_size: int = 5
    repetition_penalty: float = 1.2
    suppress_non_language: bool = True
    
    # LoRA parameters
    lora_rank: int = 32

@dataclass 
class RayClusterConfig:
    """Ray集群配置"""
    # Cluster connection
    ray_address: Optional[str] = None  # None for local mode
    redis_address: Optional[str] = None
    redis_password: Optional[str] = None
    
    # Resource allocation
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    memory: Optional[int] = None  # MB
    object_store_memory: Optional[int] = None  # MB
    
    # Ray-specific settings
    include_dashboard: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265
    temp_dir: Optional[str] = None
    
    # Logging
    log_to_driver: bool = True
    logging_level: str = "INFO"

@dataclass
class ServingConfig:
    """服务配置"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # GPU configuration
    cuda_visible_devices: str = "0,1"
    gpu_language_map: Dict[int, str] = field(default_factory=lambda: {
        0: "English -> Chinese",
        1: "English -> Italian"
    })
    
    # Batch processing
    max_batch_size: int = 32
    min_batch_size: int = 1
    batch_timeout_ms: float = 100.0
    enable_dynamic_batching: bool = True
    
    # Load balancing
    load_balance_strategy: str = "least_loaded"  # least_loaded, round_robin, gpu_memory
    
    # Session management
    session_timeout: int = 3600  # seconds
    cleanup_interval: int = 60  # seconds
    max_concurrent_sessions: int = 1000
    
    # Performance optimization
    prefetch_enabled: bool = True
    async_result_processing: bool = True
    enable_model_caching: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8001
    log_level: str = "INFO"

@dataclass
class LanguageConfig:
    """语言对配置"""
    source_lang: str
    target_lang: str
    src_code: str
    tgt_code: str
    model_config: ModelConfig
    
    def __post_init__(self):
        # 自动设置模型路径
        if not self.model_config.state_dict_path:
            if self.src_code == "en" and self.tgt_code == "de":
                self.model_config.state_dict_path = "/mnt/aries/data6/xixu/demo/en-de/pytorch_model.bin"
            elif self.src_code == "en" and self.tgt_code == "es":
                self.model_config.state_dict_path = "/mnt/aries/data6/xixu/demo/en-es/pytorch_model.bin"
            else:
                self.model_config.state_dict_path = f"/mnt/aries/data6/jiaxuanluo/demo/{self.src_code}-{self.tgt_code}/pytorch_model.bin"
        
        if not self.model_config.lora_path and self.src_code != "en" or self.tgt_code not in ["de", "es"]:
            self.model_config.lora_path = f"/mnt/aries/data6/jiaxuanluo/demo/{self.src_code}-{self.tgt_code}/lora.bin"

@dataclass
class RayInfiniSSTConfig:
    """Ray InfiniSST 系统完整配置"""
    ray_cluster: RayClusterConfig = field(default_factory=RayClusterConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    languages: Dict[str, LanguageConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        # 如果languages为空，使用默认配置
        if not self.languages:
            self.languages = self._create_default_language_configs()
    
    def _create_default_language_configs(self) -> Dict[str, LanguageConfig]:
        """创建默认的语言配置"""
        default_model_config = ModelConfig(
            model_name="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct",
            model_path="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct"
        )
        
        languages = {}
        
        # English -> Chinese
        en_zh_config = LanguageConfig(
            source_lang="English",
            target_lang="Chinese", 
            src_code="en",
            tgt_code="zh",
            model_config=ModelConfig(**asdict(default_model_config))
        )
        languages["English -> Chinese"] = en_zh_config
        
        # English -> Italian
        en_it_config = LanguageConfig(
            source_lang="English",
            target_lang="Italian",
            src_code="en", 
            tgt_code="it",
            model_config=ModelConfig(**asdict(default_model_config))
        )
        languages["English -> Italian"] = en_it_config
        
        # English -> German
        en_de_config = LanguageConfig(
            source_lang="English",
            target_lang="German",
            src_code="en",
            tgt_code="de", 
            model_config=ModelConfig(**asdict(default_model_config))
        )
        languages["English -> German"] = en_de_config
        
        # English -> Spanish
        en_es_config = LanguageConfig(
            source_lang="English",
            target_lang="Spanish",
            src_code="en",
            tgt_code="es",
            model_config=ModelConfig(**asdict(default_model_config))
        )
        languages["English -> Spanish"] = en_es_config
        
        return languages
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RayInfiniSSTConfig':
        """从配置文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RayInfiniSSTConfig':
        """从字典创建配置"""
        # 解析Ray集群配置
        ray_cluster = RayClusterConfig(**config_dict.get('ray_cluster', {}))
        
        # 解析服务配置
        serving = ServingConfig(**config_dict.get('serving', {}))
        
        # 解析语言配置
        languages = {}
        for lang_pair, lang_dict in config_dict.get('languages', {}).items():
            model_config = ModelConfig(**lang_dict.get('model_config', {}))
            lang_config = LanguageConfig(
                source_lang=lang_dict['source_lang'],
                target_lang=lang_dict['target_lang'],
                src_code=lang_dict['src_code'],
                tgt_code=lang_dict['tgt_code'],
                model_config=model_config
            )
            languages[lang_pair] = lang_config
        
        return cls(
            ray_cluster=ray_cluster,
            serving=serving,
            languages=languages
        )
    
    def to_file(self, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'ray_cluster': asdict(self.ray_cluster),
            'serving': asdict(self.serving),
            'languages': {
                lang_pair: asdict(lang_config) 
                for lang_pair, lang_config in self.languages.items()
            }
        }
    
    def get_gpu_language_map(self) -> Dict[int, str]:
        """获取GPU语言映射"""
        return self.serving.gpu_language_map
    
    def get_language_config(self, language_pair: str) -> Optional[LanguageConfig]:
        """获取指定语言对的配置"""
        return self.languages.get(language_pair)
    
    def update_from_env(self):
        """从环境变量更新配置"""
        # Ray cluster
        if os.getenv('RAY_ADDRESS'):
            self.ray_cluster.ray_address = os.getenv('RAY_ADDRESS')
        if os.getenv('RAY_NUM_CPUS'):
            self.ray_cluster.num_cpus = int(os.getenv('RAY_NUM_CPUS'))
        if os.getenv('RAY_NUM_GPUS'):
            self.ray_cluster.num_gpus = int(os.getenv('RAY_NUM_GPUS'))
        
        # Serving
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            self.serving.cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if os.getenv('MAX_BATCH_SIZE'):
            self.serving.max_batch_size = int(os.getenv('MAX_BATCH_SIZE'))
        if os.getenv('BATCH_TIMEOUT_MS'):
            self.serving.batch_timeout_ms = float(os.getenv('BATCH_TIMEOUT_MS'))
        if os.getenv('ENABLE_DYNAMIC_BATCHING'):
            self.serving.enable_dynamic_batching = os.getenv('ENABLE_DYNAMIC_BATCHING').lower() == 'true'
        
        # Update GPU language map based on CUDA_VISIBLE_DEVICES
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            gpu_ids = [int(x.strip()) for x in self.serving.cuda_visible_devices.split(',') if x.strip().isdigit()]
            available_languages = list(self.languages.keys())
            
            new_gpu_language_map = {}
            for i, gpu_id in enumerate(gpu_ids):
                if i < len(available_languages):
                    new_gpu_language_map[gpu_id] = available_languages[i]
                else:
                    new_gpu_language_map[gpu_id] = available_languages[i % len(available_languages)]
            
            self.serving.gpu_language_map = new_gpu_language_map

# ===== Utility Functions =====

def create_default_config() -> RayInfiniSSTConfig:
    """创建默认配置"""
    return RayInfiniSSTConfig()

def load_config(config_path: Optional[str] = None) -> RayInfiniSSTConfig:
    """加载配置文件，如果不存在则创建默认配置"""
    if config_path and Path(config_path).exists():
        return RayInfiniSSTConfig.from_file(config_path)
    else:
        config = create_default_config()
        config.update_from_env()
        return config

def save_default_config(config_path: str = "serve/ray_config.json"):
    """保存默认配置到文件"""
    config = create_default_config()
    config.to_file(config_path)
    print(f"默认配置已保存到: {config_path}")

# ===== Configuration Validation =====

def validate_config(config: RayInfiniSSTConfig) -> List[str]:
    """验证配置的有效性"""
    errors = []
    
    # 验证GPU配置
    gpu_ids = [int(x.strip()) for x in config.serving.cuda_visible_devices.split(',') if x.strip().isdigit()]
    if not gpu_ids:
        errors.append("No valid GPU IDs found in cuda_visible_devices")
    
    # 验证语言配置
    if not config.languages:
        errors.append("No language configurations found")
    
    for lang_pair, lang_config in config.languages.items():
        model_path = lang_config.model_config.state_dict_path
        if model_path and not Path(model_path).exists():
            errors.append(f"Model file not found for {lang_pair}: {model_path}")
        
        if lang_config.model_config.lora_path:
            lora_path = lang_config.model_config.lora_path
            if not Path(lora_path).exists():
                errors.append(f"LoRA file not found for {lang_pair}: {lora_path}")
    
    # 验证批处理配置
    if config.serving.max_batch_size < config.serving.min_batch_size:
        errors.append("max_batch_size must be greater than or equal to min_batch_size")
    
    if config.serving.batch_timeout_ms <= 0:
        errors.append("batch_timeout_ms must be positive")
    
    return errors

# ===== Example Usage =====

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray InfiniSST Configuration Management")
    parser.add_argument("--create-default", action="store_true", help="Create default configuration file")
    parser.add_argument("--config", type=str, default="serve/ray_config.json", help="Configuration file path")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.create_default:
        save_default_config(args.config)
    
    elif args.validate or args.show:
        try:
            config = load_config(args.config)
            
            if args.validate:
                errors = validate_config(config)
                if errors:
                    print("❌ Configuration validation failed:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print("✅ Configuration is valid")
            
            if args.show:
                print("\n📋 Current Configuration:")
                print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
                
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
    
    else:
        parser.print_help() 