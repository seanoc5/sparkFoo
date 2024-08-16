import org.slf4j.{Logger, LoggerFactory}
import com.johnsnowlabs.nlp.SparkNLP

object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")

    val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)

    import org.apache.spark.sql.SparkSession


    val spark = SparkSession.builder()
      .appName("Spark NLP with GPU Example")
      .master("local[*]") // Use `yarn` or `spark://...` for distributed setup
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:4.2.0")
//      .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
//      .config("spark.executor.resource.gpu.amount", "1")
//      .config("spark.executor.resource.gpu.discoveryScript", "/opt/libs/getGpusResources.sh")
//      .config("spark.task.resource.gpu.amount", "1")
//      .config("spark.sql.execution.arrow.pyspark.enabled", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")


    SparkNLP.start(gpu = true)



//    val spark = SparkSession.builder()
//      .master("local[*]")
//      .appName("Spark GPU Example")
//      .config("spark.rapids.sql.enabled", "true")
//      .config("spark.executor.resource.gpu.amount", "1")
//      .config("spark.executor.cores", "2")
//      .config("spark.task.resource.gpu.amount", "1")
//      .config("spark.rapids.sql.concurrentGpuTasks", "2")
//      .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
//      //        .config("spark.executor.extraClassPath", "/opt/libs/rapids-4-spark_2.13-24.06.1.jar")   //:/opt/libs/cudf-23.06.0-cuda11.jar
//      //      .config("spark.executor.extraJavaOptions", "-Dai.rapids.cudf.nvtx.enabled=true")
////      .config("spark.local.dir", "/tmp/spark")
//      .getOrCreate()



    // Start SparkSession using Spark NLP's utility method
    //    val spark = SparkNLP.start()
    //        val spark = SparkSession.builder.master("local[*]").appName("Spark Query Trial 1").getOrCreate()
    //    val taskGpu = spark.conf.get("spark.task.resource.gpu.amount")
    //    val execGpu = spark.conf.get("spark.executor.resource.gpu.amount")
    //    logger.info(s"Task:($taskGpu) -- Exec:(${execGpu})")

    // Check GPU configurations
    //    val foo = TaskContext.resource().get("gpu").
    println(s"GPU Amount per Task: ${spark.conf.get("spark.task.resource.gpu.amount", "Not Configured")}")
    println(s"GPU Amount per Executor: ${spark.conf.get("spark.executor.resource.gpu.amount", "Not Configured")}")


    val solrOptions = Map(
      "collection" -> "corpusminder",
      "zkhost" -> "dell:2181/solr95",
    )

    logger.info("Done?!")
  }

}
