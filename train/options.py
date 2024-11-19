def add_speech_encoder_args(parser):
    parser.add_argument(
        "--feature-extractor-cfg", 
        type=str,
        default="[(1024, 10, 5)] + [(1024, 3, 2)] * 4 + [(1024,2,2)] * 4"
    )
    parser.add_argument(
        "--feature-extractor-state-dict-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature-extractor-freeze",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--length-shrink-cfg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n-attn-layers", 
        type=int,
        default=12,
    )
    parser.add_argument(
        "--n-dim",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )    
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=12, # blocksize=1 means 80ms
    )
    parser.add_argument(
        "--max-cache-size",
        type=int, 
        default=125, # 125 * 0.08 = 1 second
    )
    parser.add_argument(
        "--llm-path", 
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lr", 
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup-updates", 
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min-lr", 
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--temp", 
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--loss-fn", 
        type=str, 
        default='waco'
    )