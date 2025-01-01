def add_speech_encoder_args(parser):
    parser.add_argument(
        "--w2v2-path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--w2v2-type",
        type=str,
        default=None
    )
    parser.add_argument(
        "--ctc-finetuned",
        action="store_true"
    )
    parser.add_argument(
        "--length-shrink-cfg",
        type=str,
        default=None,
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
        "--xpos",
        type=int,
        default=1, # 1 for True, 0 for False
    )

def add_gen_args(parser):                 
    parser.add_argument(
        "--max-len-a",
        type=int,
        default=5,
        help="Max number of tokens generated per second"
    )
    parser.add_argument(
        "--max-len-b",
        type=int,
        default=20,
        help="Max number of tokens generated additionally"
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=1
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2
    )

def add_simuleval_args(parser):
    parser.add_argument(
        "--source-lang", 
        type=str,
        default='English',
    )
    parser.add_argument(
        "--target-lang", 
        type=str,
        default='German',
    )
    parser.add_argument(
        "--min-start-sec",
        default=0.32,
        type=float,
    )