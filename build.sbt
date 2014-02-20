name := "fpis"

version := "1.0"

libraryDependencies ++= Seq(
  "org.specs2" %% "specs2" % "2.3.8" % "test",
  "org.scalaz" %% "scalaz-core" % "7.0.5"
)

scalacOptions in Test ++= Seq("-Yrangepos")

resolvers ++= Seq("snapshots", "releases").map(Resolver.sonatypeRepo)