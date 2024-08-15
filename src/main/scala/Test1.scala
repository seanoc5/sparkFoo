import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import org.apache.spark.ml.Pipeline
//import org.apache.spark.ml.Pipeline

object Test1 extends App {
  // Initialize Spark session with GPU support
  val spark = SparkSession.builder()
    .appName("Spark NLP GPU Example")
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .config("spark.jsl.settings.pretrained.cache_folder", "/tmp/spark-nlp")
    .config("spark.jsl.settings.storage.disk", "true")
    .config("spark.jsl.settings.storage.cluster_tmp_dir", "/tmp/spark-nlp")
    .getOrCreate()

  // Initialize Spark NLP
  val sparkNLP = SparkNLP.start(gpu = true)
  //  sparkNLP.set

  // Sample data

  import org.apache.spark.sql.functions._
  import spark.implicits._

  val msg = "Spark NLP is an open-source text processing library for Apache Spark."
  val data = Seq(msg).toDF("text")

  // Define the pipeline
  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

  val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

  // Fit and transform the data
  val result = pipeline.fit(data).transform(data)

  // Show the results
  result.select("token.result").show(false)

  // Stop Spark session
  spark.stop()
}
