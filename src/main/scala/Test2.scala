import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, concat_ws}
//import org.apache.spark.ml.Pipeline

object Test2 extends App {
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


  val solrOptions = Map(
    "collection" -> "corpusminder",
    "zkhost" -> "dell:2181/solr95",
    "query" -> "body_txt_en:smart",
  )

  val contentField = "body_txt_en"
  val df = spark.read.format("solr")
    .options(solrOptions)
    .option("max_rows", "100")
    .option("filters", "subtype_s:Section")
    .option("fields", "id,type_s,subtype_s,heading_txt_en,startPosition_l,title_s,body_txt_en")
    .load()

  df.show(5,80,true)

  val df2 = df.withColumn(contentField, concat_ws(" ", col(contentField)))

  // Stop Spark session
  spark.stop()
}
