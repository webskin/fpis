package sz

import scalaz._

/**
 * Created by Mickaël Gauvin on 2/20/14.
 */
object ScalazExperiments {

  def testShow() {

    /*
    scalaz/syntax/package.scala :
    ---------------------------

    package scalaz
    package syntax

    trait Syntaxes {
      …
      object show extends ToShowOps
      …
    }

    scalaz/syntax/Ops.scala :
    ------------------------------
    package scalaz.syntax

    trait Ops[A] {
      def self: A
    }


    scalaz/syntax/ShowSyntax.scala :
    ------------------------------
    package scalaz
    package syntax

    /** Wraps a value `self` and provides methods related to `Show` */
    trait ShowOps[F] extends Ops[F] {
      implicit def F: Show[F]
      ////
      final def show: Cord = F.show(self)
      final def shows: String = F.shows(self)
      final def print: Unit = Console.print(shows)
      final def println: Unit = Console.println(shows)
      ////
    }

    trait ToShowOps  {

      // ------------------------------------------------------------
      // ce qui est importé
      // ------------------------------------------------------------
      implicit def ToShowOps[F](v: F)(implicit F0: Show[F]) =
        new ShowOps[F] { def self = v; implicit def F: Show[F] = F0 }
      // ------------------------------------------------------------

      ////

      ////
    }

    trait ShowSyntax[F]  {
      implicit def ToShowOps(v: F): ShowOps[F] = new ShowOps[F] { def self = v; implicit def F: Show[F] = ShowSyntax.this.F }

      def F: Show[F]
      ////

      ////
    }

    scalaz/Show.scala :
    ------------------------------
    package scalaz

    ////
    /**
     * A typeclass for conversion to textual representation, done via
     * [[scalaz.Cord]] for efficiency.
     */
    ////
    trait Show[F]  { self =>
      ////
      def show(f: F): Cord = Cord(shows(f))
      def shows(f: F): String = show(f).toString

      def xmlText(f: F): scala.xml.Text = scala.xml.Text(shows(f))

      // derived functions
      ////
      val showSyntax = new scalaz.syntax.ShowSyntax[F] { def F = Show.this }
    }

    object Show {
      @inline def apply[F](implicit F: Show[F]): Show[F] = F

      ////

      def showFromToString[A]: Show[A] = new Show[A] {
        override def shows(f: A): String = f.toString
      }

      /** For compatibility with Scalaz 6 */
      def showA[A]: Show[A] = showFromToString[A]

      def show[A](f: A => Cord): Show[A] = new Show[A] {
        override def show(a: A): Cord = f(a)
      }

      def shows[A](f: A => String): Show[A] = new Show[A] {
        override def shows(a: A): String = f(a)
      }

      implicit def showContravariant: Contravariant[Show] = new Contravariant[Show] {
        def contramap[A, B](r: Show[A])(f: B => A): Show[B] = new Show[B] {
          override def show(b: B): Cord = r.show(f(b))
        }
      }

      ////
    }

    */

    /*

    // ------------------------------------------------------------
    // ce qui est importé
    // ------------------------------------------------------------
    implicit def ToShowOps[F](v: F)(implicit F0: Show[F]) =
        new ShowOps[F] { def self = v; implicit def F: Show[F] = F0 }

    */
    import syntax.show._
    {
      /*
      IntShow correspond à F0 dans ToShowOps ce qui revient à avoir dans le scope :
      implicit def ToShowOps[Int](v: Int) =
          new ShowOps[Int] { def self = v; implicit def F: Show[Int] = IntShow }
      */
      implicit val IntShow = new Show[Int] {
        override def shows(a: Int) = a.toString + " toto "
      }

      /*
      Du coup on a dans le scope une fonction capable de convertir un Int en ShowOps[Int]:
      implicit def ToShowOps[Int](v: Int): ShowOps[Int]
      proxifiant Show[Int] et proposant les methods : show, shows, print, println

      // Wraps a value `self` and provides methods related to `Show`
      trait ShowOps[F] extends Ops[F] {
        implicit def F: Show[F]
        ////
        final def show: Cord = F.show(self)
        final def shows: String = F.shows(self)
        final def print: Unit = Console.print(shows)
        final def println: Unit = Console.println(shows)
        ////
      }
      */
      println(3.shows)
    }

    {
      implicit val IntShow = new Show[Int] {
        override def shows(a: Int) = a.toString + " tata "
      }

      println(4.shows)

    }


  }

  def testFunctor() {

    /*
    Implicites dans le scope :

    implicit def ToFunctorOpsUnapply[FA](v: FA)(implicit F0: Unapply[Functor, FA]) =
      new FunctorOps[F0.M,F0.A] { def self = F0(v); implicit def F: Functor[F0.M] = F0.TC }

    implicit def ToFunctorOps[F[_],A](v: F[A])(implicit F0: Functor[F]) =
      new FunctorOps[F,A] { def self = v; implicit def F: Functor[F] = F0 }

    implicit def ToLiftV[F[_], A, B](v: A => B) = new LiftV[F, A, B] { def self = v }

    implicit def ToFunctorIdV[A](v: A) = new FunctorIdV[A] { def self = v }




    */

    import syntax.functor._

    // la classe Tuple de base n'a pas de methode map

    {

      /*
       scalaz/std/Tuple.scala :
      ------------------------------
      private[scalaz] trait Tuple3Functor[A1, A2] extends Traverse[({type f[x] = (A1, A2, x)})#f] {
        override def map[A, B](fa: (A1, A2, A))(f: A => B) =
          (fa._1, fa._2, f(fa._3))
        def traverseImpl[G[_], A, B](fa: (A1, A2, A))(f: A => G[B])(implicit G: Applicative[G]) =
          G.map(f(fa._3))((fa._1, fa._2, _))
      }
      */
      import std.tuple._

      val t = (1, 2, 3) map { 1 + _ }
      println(t)
    }

  }

}
