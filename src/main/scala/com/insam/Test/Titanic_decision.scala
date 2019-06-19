package com.insam.Test


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.avg
import org.apache.spark.sql.types.{IntegerType, DoubleType}


object Titanic_decision {
  def main(args: Array[String]): Unit = {

//    if (args.length < 3) {
//      System.err.println("Usage: Titanic <train file> <test file> <output file>")
//      System.exit(1)
//    }

    println("Start Decision Tree")

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Chapter 2")
      .getOrCreate()
    import spark.implicits._

    val trainFilePath = "/home/jjh/문서/dataset/titanic/train.csv"
    val titanicTrain = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(trainFilePath)
      .withColumn("Survived", $"Survived".cast(DoubleType))
      .cache()
    //titanicTrain.printSchema()
    //titanicTrain.show(5, truncate = false)
    //titanicTrain.describe("Age").show()
    //titanicTrain.describe("Fare").show()

    val avgAge = titanicTrain.select(avg("Age")).first().getDouble(0)
    val imputedTrainMap = Map[String, Any]("Age" -> avgAge, "Embarked" -> "S")
    val imputedTitanicTrain = titanicTrain.na.fill(imputedTrainMap)

    val stringCols = Seq("Sex", "Embarked")
    val indexers = stringCols.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
    }

    val numericCols = Seq("Age", "SibSp", "Parch", "Fare", "Pclass")
    val featuresCol = "features"
    val assembler = new VectorAssembler()
      .setInputCols((numericCols ++ stringCols.map(_ + "Indexed")).toArray)
      .setOutputCol(featuresCol)

    val labelCol = "Survived"
    val decisionTree = new DecisionTreeClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)

    val pipeline = new Pipeline().setStages((indexers :+ assembler :+ decisionTree).toArray)

    val model = pipeline.fit(imputedTitanicTrain)

    val testFilePath = "/home/jjh/문서/dataset/titanic/test.csv"
    val titanicTest = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(testFilePath)
      .cache()
    //titanicTest.printSchema()
    //titanicTest.show(5, truncate = false)
    //titanicTest.describe("Age").show()
    //titanicTest.describe("Fare").show()

    val avgFare = titanicTrain.select(avg("Fare")).first().getDouble(0)
    val imputedTestMap = imputedTrainMap + ("Fare" -> avgFare)
    val imputedTitanicTest = titanicTest.na.fill(imputedTestMap)

    val predictions = model.transform(imputedTitanicTest)

    val outputPath = "/home/jjh/문서/dataset/titanic/Result_DC"
    predictions
      .select($"PassengerId", $"prediction".cast(IntegerType).alias("Survived"))
      .coalesce(1)
      .write
      .option("header", "true")
      .csv(outputPath)

    spark.stop()
  }
}
