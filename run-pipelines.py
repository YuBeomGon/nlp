import argparse
import datetime
import kfp
import kfp.components as comp
from kfp import dsl
from kubernetes.client.models import V1EnvVar


@dsl.pipeline(
    name="kubeflow-petcharts-all",
    description="This is a kubeflow pipeline of petcharts",
)
def petcharts_pipeline_all(
    ACCESSKEY,
    SECRETKEY,
    BUCKET: str = "petcharts",
    CORPUSDATA: str = "corpus.txt",
    TRAINDATA: str = "train.csv",
    TESTDATA: str = "test.csv",
    TOKENIZER: str = "tokenizer.zip",
    PRETRAINED: str = "roberta.zip",
    TRANSFER: str = "roberta.transfer.zip",
    DOWNSTREAM: str = "classifier.zip",
    VOCABSIZE: int = 32000,
    CLASSES: int = 20,
    EPOCHS0: int = 24,
    EPOCHS1: int = 24,
    CONTRA_EPOCHS: int = 40,
    BATCHSIZE: int = 32,
    LOGDIR: str = "s3://petcharts/logs",
    LOGSTEPS: int = 500,
    SAVESTEPS: int = 10000,
    WEIGHTDECAY0: float = 0.1,
    WEIGHTDECAY1: float = 0.01,
    SCHEDULER0: str = "linear",
    SCHEDULER1: str = "linear",
    REGISTRYURL: str = "192.168.6.32:5000",
    HOSTURL: str = "http://minio-service.default.svc.cluster.local:9000",
):
#    tokenizer = dsl.ContainerOp(
#        name="training tokenizer",
#        image="{}/petclassify:tokenizer".format(REGISTRYURL),
#        arguments=[
#            "--host",
#            HOSTURL,
#            "--accesskey",
#            ACCESSKEY,
#            "--secretkey",
#            SECRETKEY,
#            "--bucket",
#            BUCKET,
#            "--corpusdata",
#            CORPUSDATA,
#            "--tokenizer",
#            TOKENIZER,
#            "--vocabsize",
#            VOCABSIZE,
#        ],
#    )
#    tokenizer.set_gpu_limit(1)
#    tokenizer.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
#    tokenizer.container.set_image_pull_policy("Always")

    pretrained = (
        dsl.ContainerOp(
            name="training pretrained",
            image="{}/petclassify:pretrained".format(REGISTRYURL),
            arguments=[
                "--host",
                HOSTURL,
                "--accesskey",
                ACCESSKEY,
                "--secretkey",
                SECRETKEY,
                "--bucket",
                BUCKET,
                "--corpusdata",
                CORPUSDATA,
                "--tokenizer",
                TOKENIZER,
                "--pretrained",
                PRETRAINED,
                "--vocabsize",
                VOCABSIZE,
                "--epochs",
                EPOCHS0,
                "--batchsize",
                BATCHSIZE,
                "--logdir",
                "{}.{}".format(LOGDIR, "pretrained"),
                "--logsteps",
                LOGSTEPS,
                "--savesteps",
                SAVESTEPS,
                "--weightdecay",
                WEIGHTDECAY0,
                "--scheduler",
                SCHEDULER0,
            ],
            output_artifact_paths={
                "mlpipeline-ui-metadata": "/opt/mlpipeline-ui-metadata.json"
            },
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=HOSTURL))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
        .add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=ACCESSKEY))
        .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=SECRETKEY))
    )
    pretrained.set_gpu_limit(1)
    pretrained.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
    pretrained.container.set_image_pull_policy("Always")
#    pretrained.after(tokenizer)

    downstream = (
        dsl.ContainerOp(
            name="training downstream",
            image="{}/petclassify:downstream".format(REGISTRYURL),
            arguments=[
                "--host",
                HOSTURL,
                "--accesskey",
                ACCESSKEY,
                "--secretkey",
                SECRETKEY,
                "--bucket",
                BUCKET,
                "--traindata",
                TRAINDATA,
                "--testdata",
                TESTDATA,
                "--pretrained",
                PRETRAINED,
                "--transfer",
                TRANSFER,
                "--downstream",
                DOWNSTREAM,
                "--classes",
                CLASSES,
                "--epochs",
                EPOCHS1,
                "--batchsize",
                BATCHSIZE,
                "--weightdecay",
                WEIGHTDECAY1,
                "--scheduler",
                SCHEDULER1,
                "--logdir",
                "{}.{}".format(LOGDIR, "downstream"),
            ],
            output_artifact_paths={
                "mlpipeline-ui-metadata": "/opt/mlpipeline-ui-metadata.json"
            },
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=HOSTURL))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
        .add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=ACCESSKEY))
        .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=SECRETKEY))
    )
    downstream.set_gpu_limit(1)
    downstream.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
    downstream.container.set_image_pull_policy("Always")
    downstream.after(pretrained)


@dsl.pipeline(
    name="kubeflow-petcharts-contra",
    description="This is a kubeflow pipeline of petcharts",
)
def petcharts_pipeline_contra(
    ACCESSKEY,
    SECRETKEY,
    BUCKET: str = "petcharts",
    CORPUSDATA: str = "corpus.txt",
    TRAINDATA: str = "train.csv",
    TESTDATA: str = "test.csv",
    TOKENIZER: str = "tokenizer.zip",
    PRETRAINED: str = "roberta.zip",
    TRANSFER: str = "roberta.transfer.zip",
    DOWNSTREAM: str = "classifier.zip",
    VOCABSIZE: int = 32000,
    CLASSES: int = 20,
    EPOCHS0: int = 24,
    EPOCHS1: int = 24,
    CONTRA_EPOCHS: int = 40,
    BATCHSIZE: int = 32,
    LOGDIR: str = "s3://petcharts/logs",
    LOGSTEPS: int = 500,
    SAVESTEPS: int = 10000,
    WEIGHTDECAY0: float = 0.1,
    WEIGHTDECAY1: float = 0.01,
    SCHEDULER0: str = "linear",
    SCHEDULER1: str = "linear",
    REGISTRYURL: str = "192.168.6.32:5000",
    HOSTURL: str = "http://minio-service.default.svc.cluster.local:9000",
):
    downstream = (
        dsl.ContainerOp(
            name="training contra-downstream",
            image="{}/petclassify:contra-downstream".format(REGISTRYURL),
            arguments=[
                "--host",
                HOSTURL,
                "--accesskey",
                ACCESSKEY,
                "--secretkey",
                SECRETKEY,
                "--bucket",
                BUCKET,
                "--traindata",
                TRAINDATA,
                "--testdata",
                TESTDATA,
                "--pretrained",
                PRETRAINED,
#                "--transfer",
#                TRANSFER,
                "--downstream",
                DOWNSTREAM,
                "--classes",
                CLASSES,
                "--epochs",
                EPOCHS1,
                "--contra_epochs",
                CONTRA_EPOCHS,
                "--batchsize",
                BATCHSIZE,
                "--weightdecay",
                WEIGHTDECAY1,
                "--scheduler",
                SCHEDULER1,
                "--logdir",
                "{}.{}".format(LOGDIR, "downstream"),
            ],
            output_artifact_paths={
                "mlpipeline-ui-metadata": "/opt/mlpipeline-ui-metadata.json"
            },
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=HOSTURL))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
        .add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=ACCESSKEY))
        .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=SECRETKEY))
    )
    downstream.set_gpu_limit(1)
    downstream.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
    downstream.container.set_image_pull_policy("Always")


@dsl.pipeline(
    name="kubeflow-petcharts-one",
    description="This is a kubeflow pipeline of petcharts",
)
def petcharts_pipeline_downstream(
    ACCESSKEY,
    SECRETKEY,
    BUCKET: str = "petcharts",
    CORPUSDATA: str = "corpus.txt",
    TRAINDATA: str = "train.csv",
    TESTDATA: str = "test.csv",
    TOKENIZER: str = "tokenizer.zip",
    PRETRAINED: str = "roberta.zip",
    TRANSFER: str = "roberta.transfer.zip",
    DOWNSTREAM: str = "classifier.zip",
    VOCABSIZE: int = 32000,
    CLASSES: int = 20,
    EPOCHS0: int = 24,
    EPOCHS1: int = 24,
    CONTRA_EPOCHS: int = 40,
    BATCHSIZE: int = 32,
    LOGDIR: str = "s3://petcharts/logs",
    LOGSTEPS: int = 500,
    SAVESTEPS: int = 10000,
    WEIGHTDECAY0: float = 0.1,
    WEIGHTDECAY1: float = 0.01,
    SCHEDULER0: str = "linear",
    SCHEDULER1: str = "linear",
    REGISTRYURL: str = "192.168.6.32:5000",
    HOSTURL: str = "http://minio-service.default.svc.cluster.local:9000",
):
    downstream = (
        dsl.ContainerOp(
            name="training downstream",
            image="{}/petclassify:downstream".format(REGISTRYURL),
            arguments=[
                "--host",
                HOSTURL,
                "--accesskey",
                ACCESSKEY,
                "--secretkey",
                SECRETKEY,
                "--bucket",
                BUCKET,
                "--traindata",
                TRAINDATA,
                "--testdata",
                TESTDATA,
                "--pretrained",
                PRETRAINED,
                "--transfer",
                TRANSFER,
                "--downstream",
                DOWNSTREAM,
                "--classes",
                CLASSES,
                "--epochs",
                EPOCHS1,
                "--batchsize",
                BATCHSIZE,
                "--weightdecay",
                WEIGHTDECAY1,
                "--scheduler",
                SCHEDULER1,
                "--logdir",
                "{}.{}".format(LOGDIR, "downstream"),
            ],
            output_artifact_paths={
                "mlpipeline-ui-metadata": "/opt/mlpipeline-ui-metadata.json"
            },
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=HOSTURL))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
        .add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=ACCESSKEY))
        .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=SECRETKEY))
    )
    downstream.set_gpu_limit(1)
    downstream.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
    downstream.container.set_image_pull_policy("Always")


def petcharts_pipeline_reptile(
    ACCESSKEY,
    SECRETKEY,
    BUCKET: str = "petcharts",
    CORPUSDATA: str = "corpus.txt",
    TRAINDATA: str = "unlabel_train1.csv",
    TESTDATA: str = "unlabel_test1.csv",
    TOKENIZER: str = "tokenizer.zip",
    PRETRAINED: str = "roberta.zip",
    TRANSFER: str = "roberta.transfer.zip",
    DOWNSTREAM: str = "reptile.zip",
    VOCABSIZE: int = 32000,
    CLASSES: int = 20,
    EPOCHS0: int = 24,
    EPOCHS1: int = 24,
    CONTRA_EPOCHS: int = 40,
    BATCHSIZE: int = 32,
    LOGDIR: str = "s3://petcharts/logs",
    LOGSTEPS: int = 500,
    SAVESTEPS: int = 10000,
    WEIGHTDECAY0: float = 0.1,
    WEIGHTDECAY1: float = 0.01,
    SCHEDULER0: str = "linear",
    SCHEDULER1: str = "linear",
    REGISTRYURL: str = "192.168.6.32:5000",
    HOSTURL: str = "http://minio-service.default.svc.cluster.local:9000",
):
    downstream = (
        dsl.ContainerOp(
            name="training transfer learning",
            image="{}/petclassify:reptile".format(REGISTRYURL),
            arguments=[
                "--host",
                HOSTURL,
                "--accesskey",
                ACCESSKEY,
                "--secretkey",
                SECRETKEY,
                "--bucket",
                BUCKET,
                "--pretrained",
                PRETRAINED,
                "--transfer",
                TRANSFER,
                "--downstream",
                DOWNSTREAM,
                "--classes",
                CLASSES,
                "--epochs",
                EPOCHS1,
                "--batchsize",
                BATCHSIZE,
                "--weightdecay",
                WEIGHTDECAY1,
                "--scheduler",
                SCHEDULER1,
                "--logdir",
                "{}.{}".format(LOGDIR, "downstream"),
            ],
            output_artifact_paths={
                "mlpipeline-ui-metadata": "/opt/mlpipeline-ui-metadata.json"
            },
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=HOSTURL))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
        .add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=ACCESSKEY))
        .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=SECRETKEY))
    )
    downstream.set_gpu_limit(1)
    downstream.add_node_selector_constraint("gpu-accelerator", "nvidia-highend")
    downstream.container.set_image_pull_policy("Always")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name", help="pipeline name", default="petcharts",
    )
    parser.add_argument(
        "-m", "--tags", help="tags", default=datetime.datetime.now().strftime("%Y%m%d"),
    )
    parser.add_argument("-t", "--testname", help="pipelines type", default="contra")
    parser.add_argument(
        "-H",
        "--host",
        help="kubeflow server url",
        default="http://ai.cleverai.com/pipeline/",
    )
    parser.add_argument("-A", "--accesskey", help="access key", default="haruband")
    parser.add_argument("-K", "--secretkey", help="secret key", default="haru1004")
    parser.add_argument(
        "--experiment",
        help="experiment tags",
        default=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    parser.add_argument("--vocabsize", help="vocabsize", type=int, default=32000)
    parser.add_argument("--classes", help="classes", type=int, default=15)
    parser.add_argument("--epochs0", help="pretrained epochs", type=int, default=1)
    parser.add_argument("--epochs1", help="downstream epochs", type=int, default=10)
    parser.add_argument("--contra_epochs", help="contrastive epochs", type=int, default=10)
    parser.add_argument("--batchsize", help="batchsize", type=int, default=32)
    parser.add_argument("--corpusdata", help="corpus data", default="pet_0814.txt")
#    parser.add_argument("--traindata", help="train data", default="train3.csv")
#    parser.add_argument("--testdata", help="test data", default="test3.csv")
#    parser.add_argument("--traindata", help="train data", default="disease_train.csv")
#    parser.add_argument("--testdata", help="test data", default="disease_test.csv")
#    parser.add_argument("--traindata", help="train data", default="diagcode50_train.csv")
#    parser.add_argument("--testdata", help="test data", default="diagcode50_test.csv")
    parser.add_argument("--traindata", help="train data", default="diagcode50_train1.csv")
    parser.add_argument("--testdata", help="test data", default="diagcode50_test1.csv")
    parser.add_argument(
        "--logdir", help="tensorboard logdir", default="s3://petcharts/logs"
    )
    parser.add_argument("--logsteps", help="logging steps", type=int, default=200)
    parser.add_argument("--savesteps", help="saving steps", type=int, default=10000)
    parser.add_argument(
        "--weightdecay0", help="pretrained weight decay", type=float, default=0.1
    )
    parser.add_argument(
        "--weightdecay1", help="downstream weight decay", type=float, default=0.01
    )
    parser.add_argument("--scheduler", help="scheduler type", default="linear")
    args = parser.parse_args()

    if args.testname == 'all' :
        kfp.compiler.Compiler().compile(petcharts_pipeline_all, args.name + ".zip")
    elif args.testname == 'contra':
        kfp.compiler.Compiler().compile(petcharts_pipeline_contra, args.name + ".zip")
    elif args.testname == 'down' :
        kfp.compiler.Compiler().compile(petcharts_pipeline_downstream, args.name + ".zip")
    elif args.testname == 'reptile' :
        kfp.compiler.Compiler().compile(petcharts_pipeline_reptile, args.name + ".zip")
    else :
        print('error, you should select test type well')

    client = kfp.Client()
    exp = client.create_experiment(name="{}-{}".format(args.name, args.experiment))
    run = client.run_pipeline(
        exp.id,
        args.name,
        args.name + ".zip",
        {
            "ACCESSKEY": args.accesskey,
            "SECRETKEY": args.secretkey,
            "VOCABSIZE": args.vocabsize,
            "CLASSES": args.classes,
            "EPOCHS0": args.epochs0,
            "EPOCHS1": args.epochs1,
            "CONTRA_EPOCHS": args.contra_epochs,
            "BATCHSIZE": args.batchsize,
            "LOGDIR": "{}.{}".format(args.logdir, args.tags),
            "LOGSTEPS": args.logsteps,
            "SAVESTEPS": args.savesteps,
            "WEIGHTDECAY0": args.weightdecay0,
            "WEIGHTDECAY1": args.weightdecay1,
            "SCHEDULER0": args.scheduler,
            "SCHEDULER1": args.scheduler,
            #"TOKENIZER": "tokenizer.{}.zip".format(args.tags),
            "TOKENIZER": "tokenizer.last.zip",
            #"PRETRAINED": "roberta.{}.zip".format(args.tags),
            #"PRETRAINED": "yubert.20200831.zip",
            #"PRETRAINED": "yubert.20200908.zip",
            "PRETRAINED": "roberta.20200821.zip",
            "TRANSFER": "yubert.20200908.zip",
            #"TRANSFER": "roberta.trasnfer.{}.zip".format(args.tags),
            "DOWNSTREAM": "classifier.{}.zip".format(args.tags),
            "CORPUSDATA": args.corpusdata,
            "TRAINDATA": args.traindata,
            "TESTDATA": args.testdata,
        },
    )
