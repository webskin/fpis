package sz

import scalaz._
import scala.Some

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

  scala> kind[Int]
  res0: String = Int's kind is *.
  This is a proper type. Int

  scala> kind[Option.type]
  res1: String = Option's kind is * -> *.
  This is a type constructor: a 1st-order-kinded type. Option[A]

  scala> kind[Either.type]
  res2: String = Either's kind is * -> * -> *.
  This is a type constructor: a 1st-order-kinded type. Either[L, R]

  scala> kind[Equal.type]
  res3: String = Equal's kind is * -> *.
  This is a type constructor: a 1st-order-kinded type. Equal[A]

  scala> kind[Functor.type]
  res4: String = Functor's kind is (* -> *) -> *.
  This is a type constructor that takes type constructor(s): a higher-kinded type. Functor[F[_]]
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
      println((Functor[List].compose[Option]).void(List(Some(1), None, Some(3)))) // List(Some(()), None, Some(()))
    }

    {
      // avec un implicit
      import std.list._
      import std.option._

      import syntax.functor._

      implicit val mine = Functor[List] compose Functor[Option]

      /*
      <dobblego> yeah it will always use Functor[List]
      <dobblego> notice that Functor[List ∘ Option] is actually an inline type-alias
      <dobblego> so you could wrap List ∘ Option or use the existing one
      <dobblego> I don't know of a way to easily get your desired behaviour with the raw functor composition
       */
      println(List(Some(1), None, Some(3)).void) // => problem : List((), (), ())


    }

    {
      import std.list._
      import std.option._

      import syntax.functor._

      type ListOption[α] = ({type λ[α] = List[Option[α]]})#λ[α]

      //scalaz.Functor[[α]List[Option[α]]]
      implicit val mine: Functor[ListOption] = Functor[List] compose Functor[Option]

      println(List(Some(1), None, Some(3)).void) // => non plus : List((), (), ())
    }
  }

  /*
  [[scalaz.Applicative]] without `point`.

  nouvelle méthode :
  def ap[A,B](fa: => F[A])(f: => F[A => B]): F[B]

  comparée avec :
  def map[A, B](fa: F[A])(f: A => B): F[B]

  pour `ap` fa est call by name
  ap equivaut à `<*>`

   */
  def testApply() {

    import syntax.apply._
    import std.option._
    import std.list._

    /*
    def ap[A,B](fa: => F[A])(f: => F[A => B]): F[B]

    final def <*>[B](f: F[A => B]): F[B] = F.ap(self)(f)
    F : Apply[Option]

    scalaz/std/Option.scala :
    ------------------------------

    trait OptionInstances extends OptionInstances0 {
    implicit val optionInstance = new Traverse[Option] with MonadPlus[Option] with Each[Option] with Index[Option] with Length[Option] with Cozip[Option] with Zip[Option] with Unzip[Option] with IsEmpty[Option] {
      …
      override def ap[A, B](fa: => Option[A])(f: => Option[A => B]) = f match {
        case Some(f) => fa match {
          case Some(x) => Some(f(x))
          case None    => None
        }
        case None    => None
      }
     */
    println(some(9) <*> some((_: Int) + 3)) // affiche Some(12)
    println(some(9) <*> some((_: Int) + " toto ")) // affiche Some(9 toto )


    /*
    scalaz/Bind.scala :
    ------------------------------
    trait Bind[F[_]] extends Apply[F] { self =>
      ////

      /** Equivalent to `join(map(fa)(f))`. */
      def bind[A, B](fa: F[A])(f: A => F[B]): F[B]

      override def ap[A, B](fa: => F[A])(f: => F[A => B]): F[B] = bind(f)(f => map(fa)(f))

      …
    }

    scalaz/std/List.scala :
    ------------------------------

    …
    def bind[A, B](fa: List[A])(f: A => List[B]) = fa flatMap f
    …

     */
    println(List(9, 10) <*> List((_: Int) + 3)) // List[Int] = List(12, 13)
    println(Apply[Option].ap2(some("1"), some("2"))(some((_: String) + (_: String)))) // Some(12)

    // cf. apply2
    println(^(List(9, 10), List(11, 12))((a: Int, b: Int) => a.toString + "-" + b.toString)) // List(9-11, 9-12, 10-11, 10-12)

    Apply[List].compose[Option].ap(List(Some(1), None, Some(3)))(List(Some((_: Int) + 3))) // List(Some(4), None, Some(6))

  }

  /*
  http://www.soi.city.ac.uk/~ross/papers/Applicative.pdf

  Applicative functors — an abstract characterisation of an applicative style of effectful programming,
  weaker than Monads and hence more widespread.

  From Haskell :
  Recall that Functor allows us to lift a “normal” function to a function on computational contexts.
  But fmap (map in Scala) doesn’t allow us to apply a function which is itself in a context to a value in a context.
  Applicative gives us just such a tool, (<*>).
  It also provides a method, pure, for embedding values in a default, “effect free” context.

  Laws :

  - The identity law:
  pure id <*> v = v

  - Homomorphism:
  pure f <*> pure x = pure (f x)
  Intuitively, applying a non-effectful function to a non-effectful argument in an effectful context is the same as
  just applying the function to the argument and then injecting the result into the context with pure.

  - Interchange:
  u <*> pure y = pure ($ y) <*> u
  Intuitively, this says that when evaluating the application of an effectful function to a pure argument,
  the order in which we evaluate the function and its argument doesn't matter.

  - Composition:
  u <*> (v <*> w) = pure (.) <*> u <*> v <*> w
  This one is the trickiest law to gain intuition for. In some sense it is expressing a sort of associativity
  property of (<*>). The reader may wish to simply convince themselves that this law is type-correct.

  scalaz/Applicative.scala :
  ------------------------------

  * Applicative Functor, described in [[http://www.soi.city.ac.uk/~ross/papers/Applicative.html Applicative Programming with Effects]]
  *
  * Whereas a [[scalaz.Functor]] allows application of a pure function to a value in a context, an Applicative
  * also allows application of a function in a context to a value in a context (`ap`).
  *
  * It follows that a pure function can be applied to arguments in a context. (See `map2`, `map3`, ... )
  *
  * Applicative instances come in a few flavours:
  *  - All [[scalaz.Monad]]s are also `Applicative`
  *  - Any [[scalaz.Monoid]] can be treated as an Applicative (see [[scalaz.Monoid]]#applicative)
  *  - Zipping together corresponding elements of Naperian data structures (those of of a fixed, possibly infinite shape)
  trait Applicative[F[_]] extends Apply[F] { self =>
    ////
    def point[A](a: => A): F[A]

    // alias for point
    def pure[A](a: => A): F[A] = point(a)

    // derived functions
    override def map[A, B](fa: F[A])(f: A => B): F[B] =
      ap(fa)(point(f))

    override def apply2[A, B, C](fa: => F[A], fb: => F[B])(f: (A, B) => C): F[C] =
      ap2(fa, fb)(point(f))

    …

  scalaz/std/List.scala :
  ------------------------------

  …
  //It also provides a method, pure, for embedding values in a default, “effect free” context.
  //pure takes a value of any type a, and returns a context/container of type f a. The intention is that pure creates some
  //sort of “default” container or “effect free” context. In fact, the behavior of pure is quite constrained by the laws
  //it should satisfy in conjunction with (<*>). Usually, for a given implementation of (<*>) there is only one possible
  //implementation of pure.
  def point[A](a: => A) = scala.List(a)
  …

   */
  def testApplicative() {
    import syntax.applicative._

    {
      import std.list._

      import std.option._
      import std.option.optionSyntax._
      import std.function._

      /*
      ////
      implicit def ApplicativeIdV[A](v: => A) = new ApplicativeIdV[A] {
        lazy val self = v
      }

      trait ApplicativeIdV[A] extends Ops[A] {
        def point[F[_] : Applicative]: F[A] = Applicative[F].point(self)
        def pure[F[_] : Applicative]: F[A] = Applicative[F].point(self)
        def η[F[_] : Applicative]: F[A] = Applicative[F].point(self)
      }  ////
       */
      println(1.point[List]) // List(1)

      /*
      cool about the fact that constructor is abstracted out.

      scalaz/syntax/std/OptionIdOps.scala :
      -------------------------------------

      trait OptionIdOps[A] extends Ops[A] {
        def some: Option[A] = Some(self)
      }

      trait ToOptionIdOps {
        implicit def ToOptionIdOps[A](a: A) = new OptionIdOps[A] { def self = a }
      }


       */
      println(1.point[List] map {_ + 2}) // List(3)

      println(9.some <*> {(_: Int) + 3}.some) // Some(12)

      println(1.some <* 2.some) // Some(1)

      println(none <* 2.some)  // None

      println(1.some *> 2.some) // Some(2)

      println(none *> 2.some) // None

      /*
      3.some <*> { 9.some <*> {(_: Int) + (_: Int)}.curried.some } ==
      3.some <*> {(_: Int) + 9}.some
       */
      println(3.some <*> { 9.some <*> {(_: Int) + (_: Int)}.curried.some }) // Some(12)
      println(3.some <*> {(_: Int) + 9}.some) // Some(12)

      println(^(3.some, 5.some) {_ + _}) // Some(8)
      println(^(3.some, none: Option[Int]) {_ + _}) // None

      /*
      The new ^(f1, f2) {...} style is not without the problem though. It doesn’t seem to handle Applicatives that
      takes two type parameters like Function1, Writer, and Validation. There’s another way called Applicative Builder,
      which apparently was the way it worked in Scalaz 6, got deprecated in M3, but will be vindicated again
      because of ^(f1, f2) {...}’s issues.

      scalaz/syntax/ApplySyntax.scala :
      ------------------------------
      final def |@|[B](fb: F[B]) = new ApplicativeBuilder[F, A, B] {
        val a: F[A] = self
        val b: F[B] = fb
      }

      Here’s how it looks:

       */

      println((3.some |@| 5.some) {_ + _}) // Some(8)

      println(List(9, 10) <*> List((_: Int) + 3)) // List[Int] = List(12, 13)
      println(List(9, 10) <*> List((_: Int) + 3, (_: Int) + 100)) // List(12, 13, 109, 110)

      println(^(3.some, 5.some) {_ + _}) // Some(8)
      println((3.some |@| 5.some) {_ + _}) // Some(8)

      println(^(List(9, 10), List(11, 12))((a: Int, b: Int) => a.toString + "-" + b.toString)) // List(9-11, 9-12, 10-11, 10-12)
      println((List(9, 10) |@| List(11, 12))((a: Int, b: Int) => a.toString + "-" + b.toString)) // List(9-11, 9-12, 10-11, 10-12)


      println((List(9, 10) |@| List(11, 12) |@| List(13, 14))((_, _, _))) // List((9,11,13), (9,11,14), (9,12,13), (9,12,14), (10,11,13), (10,11,14), (10,12,13), (10,12,14))

      // List[F[A]] => F[List[A]]
      def sequenceA[F[_]: Applicative, A](list: List[F[A]]): F[List[A]] = list match {
        case Nil     => (Nil: List[A]).point[F]
        case x :: xs => (x |@| sequenceA(xs)) {_ :: _}
      }

      println(sequenceA(List(1.some, 2.some))) // Some(List(1, 2))
      println(sequenceA(List(3.some, none, 1.some))) // None
      println(sequenceA(List(List(1, 2, 3), List(4, 5, 6)))) // List(List(1, 4), List(1, 5), List(1, 6), List(2, 4), List(2, 5), List(2, 6), List(3, 4), List(3, 5), List(3, 6))


      // intellij a du mal
      type Function1Int[A] = ({type l[A]=Function1[Int, A]})#l[A]
      val s = sequenceA(List((_: Int) + 3, (_: Int) + 2, (_: Int) + 1): List[Function1Int[Int]])

      println(s(3)) // List(6, 5, 4)

    }
  }

  def testMonad() {

  }
}
