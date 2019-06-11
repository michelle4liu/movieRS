import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession

object RankMethod {
  // 暂时做成   既可以处理批量用户，也可以处理单用户
  // testRatingDF 至少有三列
  // "userId","movieId","rating"
    def rank(spark : SparkSession,
             lrModel : LinearRegressionModel,
             testRatingDF : DataFrame,
             userFeatureDF : DataFrame,
             movieFeatureDF : DataFrame): DataFrame ={

    val testRatingDF2 = testRatingDF.select("userId","movieId","rating")

    //从userFeatureDF中去寻找用户的特征
    // 如果用户特征中没有这个用户的特征，那么对应的特征列中的值就为null，
    // 如果为null，放到模型中会怎么样呢？这个后续再测试
    val testRatingUser = testRatingDF2.join(userFeatureDF,userFeatureDF("userId")===testRatingDF2("userId"),"left").drop(userFeatureDF("userId"))
    val testRatingUser2 = testRatingUser.drop("zipCode","genderIndex")

    //再和movie特征结合
    val testratingUserMovie = testRatingUser2.join(movieFeatureDF,movieFeatureDF("movieId")===testRatingUser2("movieId"),"left").drop(movieFeatureDF("movieId"))
    //这个地方的movieID应该不会是空的，可以在流程上进行处理，以得到保证

    import java.util.Arrays

    //特征列是 "userId" "movieId" "rating"
    // 所以从第3列开始，即 age特征开始
    val assembler_columns = Arrays.copyOfRange(testratingUserMovie.columns,3,testratingUserMovie.columns.length)
    val t_assembler = new VectorAssembler().setInputCols(assembler_columns).setOutputCol("features")
    val t_output = t_assembler.transform(testratingUserMovie)

    // 传入一个dataSet，会自动去找“features”列, features列是一个Sparse Vector
    // 然后生成一个"prediction"列
    //预测完后还是一个DataFrame
    val prediction = lrModel.transform(t_output.select("userId","movieId","rating","features"))

    prediction
  }

  def getTopKFromDataFrame(spark : SparkSession, prediction : DataFrame): DataFrame ={
    val y = prediction
    val yRdd = y.select("userId","movieId","prediction").rdd
    val ymap2 = yRdd.map(x=>((x(0).toString.toLong,x(1).toString.toLong),(x(2).toString.toDouble)))
    val ymap3 = ymap2.groupByKey()
    val timestamp = System.currentTimeMillis()/1000
    val ymap4 = ymap3.flatMap(x=>{
      var ret : List[(Long,Long,Double,Long)] = Nil

      val key = x._1
      val IterableValue = x._2
      val sort_values = IterableValue.toList.sortWith((x,y)=>x>y).take(10)

      for (value <- sort_values){
        ret = ret ++ List((key._1,key._2,value,timestamp))
      }
      ret
    })

    import spark.implicits._
    val df_result = ymap4.toDF("userId","movieId","prediction","timestamp")

    df_result
  }

  def write2DB(resultDF : DataFrame, ip : String, port : String, database : String){
    println("======================================")
    //val url = "jdbc:mysql://192.168.199.227:3306/dataframe?characterEncoding=UTF-8"
    val url = "jdbc:mysql://" + ip + ":" + port + "/" + database + "?characterEncoding=UTF-8"
    val table="movieRS"//student表可以不存在，但是url中指定的数据库要存在
    val prop = new java.util.Properties
    prop.setProperty("user","cm")
    prop.setProperty("password","cm1q2w3e4rA!")
    prop.setProperty("host","foo-1.example.com")
    resultDF.write.mode("append").jdbc(url, table, prop)
  }
}
