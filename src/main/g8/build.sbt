import sbt.Keys.javaOptions

// give the user a nice default project!
lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.12.0",
      javaOptions += "-Xmx1G",
      libraryDependencies ++= Seq(
        "org.nd4j" % "nd4j-native-platform" % "0.8.0",
        "org.deeplearning4j" % "deeplearning4j-core" % "0.8.0",
        "org.scalatest" %% "scalatest" % "3.0.0" % "test"
      )
    )),
    name := "$name$"
  )
