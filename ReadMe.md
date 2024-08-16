# Spark Foo 
This is a simple test project in scala to explore spark config and deployment.



## Troubles & Todo
### GPU config & acceleration

#### Various Links/Sources
* https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/overview.html
* https://spark.apache.org/docs/latest/spark-standalone.html
* https://github.com/NVIDIA/spark-rapids
* https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/on-premise.html
* https://www.nvidia.com/en-us/ai-data-science/spark-ebook/gpu-accelerated-spark-3/
* https://medium.com/@gujarwasia.shubham/nvidia-gpus-and-apache-spark-a-great-combo-2f49a4b6a1ef
* https://docs.nvidia.com/spark-rapids/user-guide/23.10/getting-started/overview.html
* 
* https://docs.rapids.ai/install
* https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/abdcad308c014c0ea9554128c8beaf9030424388
* https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/Spark_NLP_Udemy_MOOC/Open_Source
* https://mvnrepository.com/artifact/ai.rapids/cudf/24.06.0
* https://docs.scala-lang.org/getting-started/intellij-track/building-a-scala-project-with-intellij-and-sbt.html
* https://www.jetbrains.com/help/idea/big-data-tools-spark.html
* https://www.nvidia.com/en-us/ai-data-science/spark-ebook/getting-started-spark-3/
* https://chatgpt.com/c/b958cc4b-876f-448b-b3a2-9aa34105b71a

#### Chat GPT & others

---------------------------- 

Certainly! Here's an example of using `spark-nlp-gpu` with GPU acceleration in Scala. This example demonstrates how to process a DataFrame with GPU-accelerated NLP tasks, such as tokenization and embeddings.

### Step 1: Set Up Dependencies

Make sure your `build.sbt` file includes the necessary dependencies for `spark-nlp-gpu` and the NVIDIA RAPIDS Spark plugin:

```sbt
libraryDependencies ++= Seq(
  "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "5.0.0",
  "com.nvidia" % "rapids-4-spark_2.12" % "23.04.0"
)
```

### Step 2: Configure Spark Session for GPU

Create a Spark session configured to use the GPU:

```scala
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.SparkNLP

val spark = SparkSession.builder()
  .appName("Spark NLP with GPU Example")
  .master("local[*]") // Use `yarn` or `spark://...` for distributed setup
  .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:5.0.0")
  .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
  .config("spark.executor.resource.gpu.amount", "1")
  .config("spark.executor.resource.gpu.discoveryScript", "/path/to/getGpusResources.sh")
  .config("spark.task.resource.gpu.amount", "1")
  .config("spark.sql.execution.arrow.pyspark.enabled", "true")
  .getOrCreate()

SparkNLP.start(spark)
```

### Step 3: Load Data and Apply NLP Pipeline

Load some text data into a DataFrame and apply a `spark-nlp` pipeline for tokenization and embeddings:

```scala
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Tokenizer, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.base.{EmbeddingsFinisher, LightPipeline}
import org.apache.spark.sql.functions._

val data = Seq(
  "Spark NLP with GPU is fast.",
  "NVIDIA GPUs accelerate NLP tasks."
).toDF("text")

// Step 4: Define the Spark NLP Pipeline
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("tokens")

val wordEmbeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
  .setInputCols("document", "tokens")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  wordEmbeddings,
  embeddingsFinisher
))

// Step 5: Train the Pipeline Model
val pipelineModel = pipeline.fit(data)

// Step 6: Apply the Pipeline Model to the DataFrame
val resultDF = pipelineModel.transform(data)

// Step 7: Display the Result
resultDF.select("finished_embeddings").show(truncate = false)
```

### Step 4: Run the Job

Ensure that your Spark cluster is correctly configured with GPUs, and then run the Scala application. The processing should be offloaded to the GPU, resulting in faster computation.

### Explanation:
1. **DocumentAssembler:** Converts the input text into a document format that can be processed by other NLP components.
2. **Tokenizer:** Splits the document into tokens.
3. **WordEmbeddingsModel:** Uses a pre-trained GloVe model to generate word embeddings.
4. **EmbeddingsFinisher:** Finalizes the embeddings so they can be used in downstream tasks.

### Notes:
- Ensure that the GPU drivers and CUDA toolkit are correctly installed.
- Modify the Spark configurations according to your environment (e.g., `yarn`, `k8s`).

This example showcases how you can use `spark-nlp-gpu` to process text data with GPU acceleration in Spark.


## Problems/Challenges
```
info] Updating 
[info] Resolved  dependencies
[warn] 
[warn]  Note: Unresolved dependencies path:
[error] sbt.librarymanagement.ResolveException: Error downloading com.johnsnowlabs.nlp:spark-nlp-gpu_2.13:5.4.1
[error]   Not found
[error]   Not found
[error]   not found: /home/sean/.ivy2/localcom.johnsnowlabs.nlp/spark-nlp-gpu_2.13/5.4.1/ivys/ivy.xml
[error]   not found: https://repo1.maven.org/maven2/com/johnsnowlabs/nlp/spark-nlp-gpu_2.13/5.4.1/spark-nlp-gpu_2.13-5.4.1.pom
[error]   not found: https://repos.spark-packages.org/com/johnsnowlabs/nlp/spark-nlp-gpu_2.13/5.4.1/spark-nlp-gpu_2.13-5.4.1.pom
[error]         at lmcoursier.CoursierDependencyResolution.unresolvedWarningOrThrow(CoursierDependencyResolution.scala:344)
[error]         at lmcoursier.CoursierDependencyResolution.$anonfun$update$38(CoursierDependencyResolution.scala:313)
[error]         at scala.util.Either$LeftProjection.map(Either.scala:573)
[error]         at lmcoursier.CoursierDependencyResolution.update(CoursierDependencyResolution.scala:313)
[error]         at sbt.librarymanagement.DependencyResolution.update(DependencyResolution.scala:60)
[error]         at sbt.internal.LibraryManagement$.resolve$1(LibraryManagement.scala:60)
[error]         at sbt.internal.LibraryManagement$.$anonfun$cachedUpdate$12(LibraryManagement.scala:134)
[error]         at sbt.util.Tracked$.$anonfun$lastOutput$1(Tracked.scala:74)
[error]         at sbt.internal.LibraryManagement$.$anonfun$cachedUpdate$20(LibraryManagement.scala:147)
[error]         at scala.util.control.Exception$Catch.apply(Exception.scala:228)
[error]         at sbt.internal.LibraryManagement$.$anonfun$cachedUpdate$11(LibraryManagement.scala:147)
[error]         at sbt.internal.LibraryManagement$.$anonfun$cachedUpdate$11$adapted(LibraryManagement.scala:128)
[error]         at sbt.util.Tracked$.$anonfun$inputChangedW$1(Tracked.scala:220)
[error]         at sbt.internal.LibraryManagement$.cachedUpdate(LibraryManagement.scala:161)
[error]         at sbt.Classpaths$.$anonfun$updateTask0$1(Defaults.scala:3801)
[error]         at scala.Function1.$anonfun$compose$1(Function1.scala:49)
[error]         at sbt.internal.util.$tilde$greater.$anonfun$$u2219$1(TypeFunctions.scala:63)
[error]         at sbt.std.Transform$$anon$4.work(Transform.scala:69)
[error]         at sbt.Execute.$anonfun$submit$2(Execute.scala:283)
[error]         at sbt.internal.util.ErrorHandling$.wideConvert(ErrorHandling.scala:24)
[error]         at sbt.Execute.work(Execute.scala:292)
[error]         at sbt.Execute.$anonfun$submit$1(Execute.scala:283)
[error]         at sbt.ConcurrentRestrictions$$anon$4.$anonfun$submitValid$1(ConcurrentRestrictions.scala:265)
[error]         at sbt.CompletionService$$anon$2.call(CompletionService.scala:65)
[error]         at java.base/java.util.concurrent.FutureTask.run(FutureTask.java:264)
[error]         at java.base/java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:515)
[error]         at java.base/java.util.concurrent.FutureTask.run(FutureTask.java:264)
[error]         at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
[error]         at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
[error]         at java.base/java.lang.Thread.run(Thread.java:829)
[error] (update) sbt.librarymanagement.ResolveException: Error downloading com.johnsnowlabs.nlp:spark-nlp-gpu_2.13:5.4.1
[error]   Not found
[error]   Not found
[error]   not found: /home/sean/.ivy2/localcom.johnsnowlabs.nlp/spark-nlp-gpu_2.13/5.4.1/ivys/ivy.xml
[error]   not found: https://repo1.maven.org/maven2/com/johnsnowlabs/nlp/spark-nlp-gpu_2.13/5.4.1/spark-nlp-gpu_2.13-5.4.1.pom
[error]   not found: https://repos.spark-packages.org/com/johnsnowlabs/nlp/spark-nlp-gpu_2.13/5.4.1/spark-nlp-gpu_2.13-5.4.1.pom
[error] Total time: 2 s, completed Aug 15, 2024, 10:40:16 PM
```
