ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "sparkFoo2"
  )

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.1"
//libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "4.2.0"
// https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp-gpu
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "5.4.1"

libraryDependencies ++= Seq(
  "com.lucidworks.spark" % "spark-solr" % "4.0.4",
)

// Add the Spark NLP repository
resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"
