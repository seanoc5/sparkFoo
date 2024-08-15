ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "sparkFoo2"
  )

//libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.1"
//libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.1"
//libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.1"

// https://mvnrepository.com/artifact/com.nvidia/rapids-4-spark
//libraryDependencies += "com.nvidia" %% "rapids-4-spark" % "24.08.0"
// https://mvnrepository.com/artifact/com.nvidia/rapids-4-spark
//libraryDependencies += "com.nvidia" %% "rapids-4-spark" % "24.06.1"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-sql" % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0",

//  "com.johnsnowlabs.nlp" %% "spark-nlp" % "4.2.0",
  "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "4.2.0"
    // https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
//    libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "5.4.0"


)

libraryDependencies ++= Seq(
//  "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "5.0.0",
  "com.nvidia" % "rapids-4-spark_2.12" % "23.04.0"
)

// Add the Spark NLP repository
resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"
