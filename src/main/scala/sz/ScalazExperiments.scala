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
      println(3.shows) // affiche 3 toto
    }

    {
      implicit val IntShow = new Show[Int] {
        override def shows(a: Int) = a.toString + " tata "
      }

      println(4.shows) // affiche 4 tata

    }


  }

  /*
  From Haskell :
  The Functor class (haddock) is the most basic and ubiquitous type class in the Haskell libraries.
  A simple intuition is that a Functor represents a “container” of some sort, along with the ability to apply a
  function uniformly to every element in the container. For example, a list is a container of elements,
  and we can apply a function to every element of a list, using map. As another example, a binary tree is also a
  container of elements, and it’s not hard to come up with a way to recursively apply a function to every element in a tree.

  Another intuition is that a Functor represents some sort of “computational context”.
  This intuition is generally more useful, but is more difficult to explain, precisely because it is so general.
  Some examples later should help to clarify the Functor-as-context point of view.

  In the end, however, a Functor is simply what it is defined to be;
  doubtless there are many examples of Functor instances that don’t exactly
  fit either of the above intuitions. The wise student will focus their attention on definitions and examples,
  without leaning too heavily on any particular metaphor. Intuition will come, in time, on its own.

  Lois :
  trait FunctorLaw {
    /** The identity function, lifted, is a no-op. */
    def identity[A](fa: F[A])(implicit FA: Equal[F[A]]): Boolean = FA.equal(map(fa)(x => x), fa)

    /**
     * A series of maps may be freely rewritten as a single map on a
     * composed function.
     */
    def composite[A, B, C](fa: F[A], f1: A => B, f2: B => C)(implicit FC: Equal[F[C]]): Boolean = FC.equal(map(map(fa)(f1))(f2), map(fa)(f2 compose f1))
  }
  The first law says that mapping the identity function over every item in a container has no effect.
  The second says that mapping a composition of two functions over every item in a container is the same as first
  mapping one function, and then mapping the other.
  */
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
      println(t) // affiche (1, 2, 4)    (fa._1, fa._2, f(fa._3))
    }

    {
      // Composition
      // ----------------

      // import functor instances
      import std.option._

      /*
      scalaz/std/List.scala :
      ------------------------------
      …
      // Traverse is a functor
      trait ListInstances extends ListInstances0 {
        // Le functor implicit
        implicit val listInstance = new Traverse[List] with MonadPlus[List] with Each[List] with Index[List] with Length[List] with Zip[List] with Unzip[List] with IsEmpty[List] {
      …
      object list extends ListInstances with ListFunctions {
        object listSyntax extends scalaz.syntax.std.ToListOps
      }
      …
      */
      import std.list._


      /*

      scalaz/Functor.scala :
      ------------------------------

      // The composition of Functors `F` and `G`, `[x]F[G[x]]`, is a Functor
      def compose[G[_]](implicit G0: Functor[G]): Functor[({type λ[α] = F[G[α]]})#λ] = new CompositionFunctor[F, G] {
        implicit def F = self

        implicit def G = G0
      }
      …
      object Functor {
        @inline def apply[F[_]](implicit F: Functor[F]): Functor[F] = F
      }

      scalaz/Composition.scala :
      ------------------------------

      private[scalaz] trait CompositionFunctor[F[_], G[_]] extends Functor[({type λ[α] = F[G[α]]})#λ] {
        implicit def F: Functor[F]

        implicit def G: Functor[G]

        override def map[A, B](fga: F[G[A]])(f: A => B): F[G[B]] = F(fga)(ga => G(ga)(f))
      }

      Functor[List] compose Functor[Option] ==
      Functor[List].apply[List[_]](listInstance) compose Functor[Option].apply[Option[_]](optionInstance)  ==
      listInstance compose optionInstance


      In repl :
      scala> Functor[List] compose Functor[Option]
      res1: scalaz.Functor[[α]List[Option[α]]] = scalaz.Functor$$anon$1@598e89f5

      scala> :t res1
      scalaz.Functor[[α]List[Option[α]]]
      */
      val compose1 = Functor[List] compose Functor[Option] // scalaz.Functor[[α]List[Option[α]]]

      /*
      map[Int, Double](fga: List[Option[Int]])(f: Int => Double): List[Option[Double]] ==
      listInstance.apply(fga)(ga => optionInstance.apply(ga)(f)) ==
      listInstance.map(fga)(ga => optionInstance.map(ga)(f)) ==
      listInstance.traversal[Id](Id.id).run(fga)(ga => ga map f) ==
      listInstance.traverseImpl[Id,Int,Double](fga)(ga => ga map f)(Id.id) ==

      DList.fromList(fga).foldr(Id.id.point(Option[Double]())) {
         (ga, fbs) => Id.id.apply2(ga => ga map f, fbs)(_ :: _)
      }

      ==

      DList.fromList(fga: List[Option[Int]]).foldr(List[Option[Double]]()) {
         (ga, fbs) => ap(fbs)(map(ga => ga map f)((_ :: _).curried))
      }

      */
      val fga = List(Some(1), None, Some(3))
      val transfo: Int => Double = _ / 2.0
      println(compose1.map(fga)(transfo)) // affiche List(Some(0.5), None, Some(1.5))
      println(listInstance.map(fga)(ga => optionInstance.apply(ga)(transfo))) // affiche List(Some(0.5), None, Some(1.5))

      {
        import Id.Id
        println(listInstance.traverseImpl[Id,Option[Int],Option[Double]](fga)(ga => ga map transfo)(Id.id)) // affiche List(Some(0.5), None, Some(1.5))
      }


      println(intersperse(List(Some(1), Some(2), Some(3)), None)) // List(Some(1), None, Some(2), None, Some(3))
      println(toNel(List(Some(1), None, Some(3)))) // affiche Some(NonEmptyList(Some(1), None, Some(3)))



      // Product
      // ---------------------

      /*
      private[scalaz] trait ProductFunctor[F[_], G[_]] extends Functor[({type λ[α] = (F[α], G[α])})#λ] {
        implicit def F: Functor[F]

        implicit def G: Functor[G]

        override def map[A, B](fa: (F[A], G[A]))(f: A => B): (F[B], G[B]) = (F.map(fa._1)(f), G.map(fa._2)(f))
      }
      */
      val product1 = Functor[List] product Functor[Option] // scalaz.Functor[[α](List[α], Option[α])]

      println(product1.map((List(1, 2, 3), Some(4)))(_.toString + " titi ")) // affiche (List(1 titi , 2 titi , 3 titi ),Some(4 titi ))

      val product2 = compose1 product Functor[Option] // scalaz.Functor[[α](List[Option[α]], Option[α])]

      println(product2.map((List(Some(1), Some(2), Some(3), None), Some(4)))(_.toString + " titi ")) // affiche (List(Some(1 titi ), Some(2 titi ), Some(3 titi ), None),Some(4 titi ))

      // Lift
      // ---------------------

      val lifted = product2.lift(transfo) //  ((List[Option[Int]], Option[Int])) => (List[Option[Double]], Option[Double])

      println(lifted((List(Some(1), Some(2), Some(3), None), Some(4)))) // (List(Some(0.5), Some(1.0), Some(1.5), None),Some(2.0))


      // strengthL
      // ---------------------
      // compose1 = scalaz.Functor[[α]List[Option[α]]]
      println(compose1.strengthL(111, fga)) // List(Some((111,1)), None, Some((111,3)))

    }

    {

      import std.list._
      import std.option._

      // mapply
      // ---------------------
      val transfoA: Int => Double = _ / 2.0
      val transfoB: Int => Double = _ + 1
      val transfoC: Int => Double = _ * 2
      val functions = List(transfoA, transfoB, transfoC)

      println(Functor[List].mapply(10)(functions)) // List[Double] = List(5.0, 11.0, 20.0)

      // void
      // ---------------------
      println((Functor[List] compose Functor[Option]).void(List(Some(1), None, Some(3)))) // List(Some(()), None, Some(()))
    }

  }

  /*
  [[scalaz.Applicative]] without `point`.

  nouvelle méthode :
  def ap[A,B](fa: => F[A])(f: => F[A => B]): F[B]

  comparée avec :
  def map[A, B](fa: F[A])(f: A => B): F[B]

  pour `ap` fa est call by name

   */
  def testApply() {

    import syntax.apply._


  }
}
