import os
import argparse
import json
import torch
import numpy as np
from minioset import (
    connect_server,
    load_object,
    save_object,
    compress_object,
    uncompress_object,
)
from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from modeling_yubert import YubertForMaskedLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--bucket", help="bucket name", default="petcharts")
    parser.add_argument("-c", "--corpusdata", help="corpus file", default="pet_wiki.txt")
    parser.add_argument(
        "-k", "--tokenizer", help="tokenizer zip file", default="tokenizer.zip"
    )
    parser.add_argument(
        "-p", "--pretrained", help="pretrained model zip file", default="roberta.zip"
    )
    parser.add_argument("-v", "--vocabsize", help="vocabsize", type=int, default=40000)
    parser.add_argument("-e", "--epochs", help="epochs", type=int, default=24)
    parser.add_argument("-b", "--batchsize", help="batchsize", type=int, default=32)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-H", "--host", help="object server")
    parser.add_argument("-A", "--accesskey", help="access key")
    parser.add_argument("-K", "--secretkey", help="secret key")
    parser.add_argument("--logdir", help="tensorboard logdir", default="./logs")
    parser.add_argument("--logsteps", help="logging steps", type=int, default=500)
    parser.add_argument("--warmupsteps", help="warmup steps", type=int, default=90000)
    parser.add_argument("--savesteps", help="saving steps", type=int, default=10000)
    parser.add_argument("--weightdecay", help="weight decay", type=float, default=0.1)
    parser.add_argument("--scheduler", help="scheduler type", default="linear")
    args = parser.parse_args()
    print(args.pretrained)

    try:
        client = connect_server(args.host, args.accesskey, args.secretkey)
        load_object(client, args.bucket, args.corpusdata)
        load_object(client, args.bucket, args.tokenizer)
        load_object(client, args.bucket, args.pretrained)
    except:
        pass

    try:
        uncompress_object(args.tokenizer, ".")
        uncompress_object(args.pretrained, ".")
    except:
        pass

    tokenizer = RobertaTokenizerFast.from_pretrained("./pretrained", max_len=512)

    config = RobertaConfig(
        vocab_size=args.vocabsize,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        isjupyter=False
    )

    device = torch.device(args.device)
    try:
        model = YubertForMaskedLM.from_pretrained("./pretrained", config=config)
    except:
        model = YubertForMaskedLM(config=config)
    model.to(device)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=args.corpusdata, block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    SEED = np.random.randint(0, 100000, size=None)

    training_args = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_gpu_train_batch_size=args.batchsize,
        logging_dir=args.logdir,
        logging_steps=args.logsteps,
        logging_first_step=False,
        #max_steps=20100,
        save_steps=args.savesteps,
        save_total_limit=5,
        seed=SEED
    )

    optimizer = AdamW(
        model.parameters(), lr=0.00006, betas=(0.9, 0.999), weight_decay=args.weightdecay
    )
    if args.scheduler == "cosine":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmupsteps,
            num_training_steps=len(dataset) * args.epochs,
            num_cycles=1.0,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmupsteps,
            num_training_steps=len(dataset) * args.epochs,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    trainer.save_model("./pretrained")

    compress_object(args.pretrained, "./pretrained")

    try:
        save_object(client, args.bucket, args.pretrained)
    except :
        print("*****************model save error to minio*******************")
        #pass

    metadata = {"outputs": [{"type": "tensorboard", "source": args.logdir}]}
    with open("/opt/mlpipeline-ui-metadata.json", "w") as fd:
        json.dump(metadata, fd)
