package com.insam.Mllib

import org.apache.spark.{SparkConf, SparkContext}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.DenseMatrix
import breeze.linalg.csvwrite
import java.io.File

object mcSpark_7_SvdPca {
  def main(args: Array[String]): Unit = {
    val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
    val sc= new SparkContext(conf)


    // ########################################################################## 1
    /*
      Ensure you you start your Spark shell with enough memory:
        ./bin/spark-shell --driver-memory 2g
    */

    /* Replace 'PATH' with the path to the LFW data */
    val path = "D:\\lfw\\*"
    val rdd = sc.wholeTextFiles(path) // 하나의 파일이 하나의 파티션에 들어가도록 하는 명령어
    val first = rdd.first
    println(first)
    // first: (String, String) =  (file:/PATH/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg,??????JFIF????? ...


    // extract just the file names
    val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
    println(files.first)
    // file:/PATH/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
    println(files.count)
    /*
    ..., /PATH/lfw/Azra_Akin/Azra_Akin_0003.jpg:0+19927, /PATH/lfw/Azra_Akin/Azra_Akin_0004.jpg:0+16030
    1055
    */
    // ########################################################################## 3
    // load an image from a file

    // 데이터브릭스에서 제공해주는것으로는 텐서프레임으로 이미지를 바로 데이터프레임으로 읽을 수 있다.
    def loadImageFromFile(path: String): BufferedImage = {

      ImageIO.read(new File(path))
    }

    val aePath = "D:\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg"
    val aeImage = loadImageFromFile(aePath)
    /*
    aeImage: java.awt.image.BufferedImage = BufferedImage@f41266e: type = 5 ColorModel: #pixelBits = 24
      numComponents = 3 color space = java.awt.color.ICC_ColorSpace@7e420794 transparency = 1 has alpha = false
      isAlphaPre = false ByteInterleavedRaster: width = 250 height = 250 #numDataElements 3 dataOff[0] = 2
    */

    // convert an image to grayscale, and scale it to new width and height
    // 원래 이미지는 250픽셀인데, rgb 중에서 색깔을 null로 바꿔서 회색으로 바꾼다
    def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
      val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      val g = bwImage.getGraphics()
      g.drawImage(image, 0, 0, width, height, null)
      g.dispose()
      bwImage
    }

    // 그리고 100*100 으로 바꾼다(압축)
    val grayImage = processImage(aeImage, 100, 100)
    /*
    grayImage: java.awt.image.BufferedImage = BufferedImage@21f8ea3b: type = 10 ColorModel: #pixelBits = 8
      numComponents = 1 color space = java.awt.color.ICC_ColorSpace@5cd9d8e9 transparency = 1 has alpha = false
      isAlphaPre = false ByteInterleavedRaster: width = 100 height = 100 #numDataElements 1 dataOff[0] = 0
    */
    // write the image out to the file system

    //. jpg로 저장
    ImageIO.write(grayImage, "jpg", new File("D:\\tmp\\aeGray.jpg"))


    // #######################################################################  5
    // extract the raw pixels from the image as a Double array
    def getPixelsFromImage(image: BufferedImage): Array[Double] = {
      val width = image.getWidth
      val height = image.getHeight
      val pixels = Array.ofDim[Double](width * height)
      image.getData.getPixels(0, 0, width, height, pixels)
      // pixels.map(p => p / 255.0) 		// optionally scale to [0, 1] domain
    }

    // put all the functions together
    def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
      val raw = loadImageFromFile(path)
      val processed = processImage(raw, width, height)
      getPixelsFromImage(processed)
    }

    // 결국 1024개의 이미지를 한 행당 2,500개씩 픽셀로 넣어서 만든다
    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))
    /*
    0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0, ...
    241.0,243.0,245.0,244.0,231.0,205.0,177.0,160.0,150.0,147.0, ...
    253.0,253.0,253.0,253.0,253.0,253.0,254.0,254.0,253.0,253.0, ...
    244.0,244.0,243.0,242.0,241.0,240.0,239.0,239.0,237.0,236.0, ...
    44.0,47.0,47.0,49.0,62.0,116.0,173.0,223.0,232.0,233.0, ...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, ...
    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0, ...
    26.0,26.0,27.0,26.0,24.0,24.0,25.0,26.0,27.0,27.0, ...
    240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0, ...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, ...
    */

    // create vectors
    // 덴스 벡터로 변환
    val vectors = pixels.map(p => Vectors.dense(p))
    // the setName method createa a human-readable name that is displayed in the Spark Web UI
    vectors.setName("image-vectors")
    // remember to cache the vectors to speed up computation
    vectors.cache

    // ########################################################################## 6
    // normalize the vectors by subtracting the column means
    // 평균/표준편차를 통해 표준화 시킨다
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))


    // ########################################################################## 7
    // create distributed RowMatrix from vectors, and train PCA on it

    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)
    // you may see warnings, if the native BLAS libraries are not installed, don't worry about these
    // 14/09/17 19:53:49 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    // 14/09/17 19:53:49 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK


    // use Breeze to save the principal components as a CSV file
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)
    // (2500,10) 2500개의 픽셀에서 10개의 성분(고유벡터?)


    // ########################################################################### 8
    // 덴스벡터로 만들자(대부분 데이터가 포함되어있으니까?)
    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
        // 2500,10개짜리

    csvwrite(new File("D:\\tmp\\pc.csv"), pcBreeze)


    // project the raw images to the K-dimensional space of the principla components
    val projected = matrix.multiply(pc)
    println(projected.numRows, projected.numCols)
    // (1054,10)
    println(projected.rows.take(5).mkString("\n"))
    /*
    [2648.9455749636277,1340.3713412351376,443.67380716760965,-353.0021423043161,52.53102289832631,423.39861446944354,413.8429065865399,-484.18122999722294,87.98862070273545,-104.62720604921965]
    [172.67735747311974,663.9154866829355,261.0575622447282,-711.4857925259682,462.7663154755333,167.3082231097332,-71.44832640530836,624.4911488194524,892.3209964031695,-528.0056327351435]
    [-1063.4562028554978,388.3510869550539,1508.2535609357597,361.2485590837186,282.08588829583596,-554.3804376922453,604.6680021092125,-224.16600191143075,-228.0771984153961,-110.21539201855907]
    [-4690.549692385103,241.83448841252638,-153.58903325799685,-28.26215061165965,521.8908276360171,-442.0430200747375,-490.1602309367725,-456.78026845649435,-78.79837478503592,70.62925170688868]
    [-2766.7960144161225,612.8408888724891,-405.76374113178616,-468.56458995613974,863.1136863614743,-925.0935452709143,69.24586949009642,-777.3348492244131,504.54033662376435,257.0263568009851]
    */

    // ######################################################################### 10
    // relationship to SVD
    // 3개로 인수분해 => pca와는 다르게 원래 이미지로 복원 가능, pca는 복원 불가능
    val svd = matrix.computeSVD(10, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
    // U dimension: (1055, 10) : 행 기준 인수분해
    // S dimension: (10, 2500)
    // V dimension: (2500, 10) : 열 기준 인수분해


    // ######################################################################### 11
    // simple function to compare the two matrices, with a tolerance for floating point number comparison
    def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
      // note we ignore sign of the principal component / singular vector elements
      val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true }
      bools.fold(true)(_ & _)
    }
    // test the function
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
    // true
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))
    // false
    println(approxEqual(svd.V.toArray, pc.toArray))
    // true




    // ######################################################################## 12
    // compare projections
    // svd의 u와s 행렬곱
    val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
    val projectedSVD = svd.U.rows.map { v =>
      val breezeV = breeze.linalg.DenseVector(v.toArray)
      val multV = breezeV :* breezeS
      Vectors.dense(multV.data)
    }
    //pca의 projected
    //pca와 u*v를 같은지 판단 => 동일하다
    projected.rows.zip(projectedSVD).map { case (v1, v2) => approxEqual(v1.toArray, v2.toArray) }.filter(b => true).count
    // 1054
    /* pca와 svd의 차이점
    복원이 필요할때만 svd쓴다.
    pca를 통해 필요한 성분을 압축해서, 학습을 하면 더 효과적이다.
    그런데 요즘은 성분을 압축하지 않고 학습하는게 딥러닝?
     */


    // ######################################################################### 13
    // 그러면 몇개로 주성분을 묶으면 될까?

    // inspect singular values
    val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU = false).s }
    sValues.foreach(println)
    /*
    데이터를 잘 표현하는 고유벡터들
    [54091.00997110354]
    [54091.00997110358,33757.702867982436]
    [54091.00997110357,33757.70286798241,24541.193694775946]
    [54091.00997110358,33757.70286798242,24541.19369477593,23309.58418888302]
    [54091.00997110358,33757.70286798242,24541.19369477593,23309.584188882982,21803.09841158358]
    */
    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
    csvwrite(new File("D:\\tmp\\s.csv"), sMatrix)

  }
}
