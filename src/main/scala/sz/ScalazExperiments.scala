package sz

import scalaz._

/**
 * Created by Mickaël Gauvin on 2/20/14.
 */
object ScalazExperiments {

  def testShow() {

    {
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
        implicit def ToShowOps[F](v: F)(implicit F0: Show[F]) =
          new ShowOps[F] { def self = v; implicit def F: Show[F] = F0 }

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
      import syntax.show._

      implicit val IntShow = new Show[Int] {
        override def shows(a: Int) = a.toString
      }

      println(3.shows)
      println(3.show)

    }



  }

}
