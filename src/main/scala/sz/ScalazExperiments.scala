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

    import std.anyVal._ // default shows

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

    {
      import std.anyVal._

      println(4.shows) // affiche 4
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

  /*
  LYAHFGG:

  It seems that both * together with 1 and ++ along with [] share some common properties: - The function takes two
  parameters. - The parameters and the returned value have the same type. - There exists such a value that doesn’t
  change other values when used with the binary function.

  It doesn’t matter if we do (3 * 4) * 5 or 3 * (4 * 5). Either way, the result is 60. The same goes for ++. …
  We call this property associativity. * is associative, and so is ++, but -, for example, is not.

  scala> (3 * 2) * (8 * 5) assert_=== 3 * (2 * (8 * 5))

  scala> List("la") ++ (List("di") ++ List("da")) assert_=== (List("la") ++ List("di")) ++ List("da")

  A monoid is when you have an associative binary function and a value which acts as an identity with respect to that
  function.

  // An associative binary operation, circumscribed by type and the
  // semigroup laws.  Unlike [[scalaz.Monoid]], there is not necessarily
  // a zero.
  trait Semigroup[A]  { self =>
    def append(a1: A, a2: => A): A
    ...
  }

  // Provides an identity element (`zero`) to the binary `append`
  // operation in [[scalaz.Semigroup]], subject to the monoid laws.
  //
  // Example instances:
  //  - `Monoid[Int]`: `zero` and `append` are `0` and `Int#+` respectively
  //  - `Monoid[List[A]]`: `zero` and `append` are `Nil` and `List#++` respectively
  // LYAHFGG:
  //
  // mempty represents the identity value for a particular monoid.
  //
  // Scalaz calls this zero instead.
  trait Monoid[A] extends Semigroup[A] { self =>
    ////
    // The identity element for `append`.
    def zero: A

    ...
  }

  // We have mappend, which, as you’ve probably guessed, is the binary function. It takes two values of the
  // same type and returns a
  // value of that type as well.
  trait SemigroupOps[A] extends Ops[A] {
    final def |+|(other: => A): A = A.append(self, other)
    final def mappend(other: => A): A = A.append(self, other)
    final def ⊹(other: => A): A = A.append(self, other)
  }

  cf. http://eed3si9n.com/learning-scalaz/Monoid.html

  Semigroup and Monoid as a way of abstracting binary operations over various types.

   */
  def testMonoid() {

    {
      import syntax.semigroup._
      import syntax.order._
      import std.list._
      import std.string._
      import std.option._
      import std.option.optionSyntax._
      println(List(1, 2, 3) |+| List(4, 5, 6)) // List(1, 2, 3, 4, 5, 6)
      println("one" |+| "two") // onetwo

      println(Monoid[List[Int]].zero) // List()
      println(Monoid[String].zero) // ""

      def lengthCompare(lhs: String, rhs: String): Ordering =
        (lhs.length ?|? rhs.length) |+| (lhs ?|? rhs)

      println(lengthCompare("zen", "ants")) // LT
      println(lengthCompare("zen", "ant")) // GT


      // cf. http://eed3si9n.com/learning-scalaz/Option+as+Monoid.html
      println((none: Option[String]) |+| "andy".some) // Some(andy)
      println("toto".some |+| "andy".some) // Some(totoandy)
    }
  }

  /*
  From Haskell :
  Monads are a natural extension applicative functors, and they provide a solution to the following problem:
  If we have a value with context, m a, how do we apply it to a function that takes a normal a
  and returns a value with a context.

  flatMap

  trait Monad[F[_]] extends Applicative[F] with Bind[F] { self =>
    ////
  }

  trait Bind[F[_]] extends Apply[F] { self =>
    //  Equivalent to `join(map(fa)(f))`.   join <=> flatten
    def bind[A, B](fa: F[A])(f: A => F[B]): F[B]
  }

  // Wraps a value `self` and provides methods related to `Bind`
  trait BindOps[F[_],A] extends Ops[F[A]] {
    implicit def F: Bind[F]
    ////
    import Liskov.<~<

    def flatMap[B](f: A => F[B]) = F.bind(self)(f)
    def >>=[B](f: A => F[B]) = F.bind(self)(f)
    def ∗[B](f: A => F[B]) = F.bind(self)(f)
    def join[B](implicit ev: A <~< F[B]): F[B] = F.bind(self)(ev(_))
    def μ[B](implicit ev: A <~< F[B]): F[B] = F.bind(self)(ev(_))
    def >>[B](b: F[B]): F[B] = F.bind(self)(_ => b)
    def ifM[B](ifTrue: => F[B], ifFalse: => F[B])(implicit ev: A <~< Boolean): F[B] = {
      val value: F[Boolean] = Liskov.co[F, A, Boolean](ev)(self)
      F.ifM(value, ifTrue, ifFalse)
    }
    ////
  }
   */
  def testMonad() {

    import syntax.monad._
    import std.list._
    import std.string._
    import std.option._
    import std.option.optionSyntax._


    println(Monad[Option].point("WHAT")) // Some(WHAT)
    println(9.some flatMap { x => Monad[Option].point(x * 10) }) // Some(90)

    /*
    Let’s say that [Pierre] keeps his balance if the number of birds on the left side of the pole and on the right side
    of the pole is within three. So if there’s one bird on the right side and four birds on the left side, he’s okay.
    But if a fifth bird lands on the left side, then he loses his balance and takes a dive.
    We may also devise a function that ignores the current number of birds on the balancing pole and just
    makes Pierre slip and fall. We can call it banana.
     */
    type Birds = Int
    case class Pole(left: Birds, right: Birds) {
      def landLeft(n: Birds): Option[Pole] = {
        println("landLeft :" + n)
        if (math.abs((left + n) - right) < 4) copy(left = left + n).some
        else none
      }
      def landRight(n: Birds): Option[Pole] = {
        println("landRight :" + n)
        if (math.abs(left - (right + n)) < 4) copy(right = right + n).some
        else none
      }
      def banana: Option[Pole] = {
        println("banana :")
        none
      }
    }

    // >>=   is flatMap
    // prints
    /*
    landLeft :1
    landRight :4
    landLeft :-1
    None
     */
    println(Monad[Option].point(Pole(0, 0)) >>= {_.landLeft(1)} >>= {_.landRight(4)} >>= {_.landLeft(-1)} /* fails fast */ >>= {_.landRight(-2)} >>= {_.landRight(-2)})

    // Option is already a monad
    // prints
    /*
    landLeft :1
    landRight :4
    landLeft :-1
    None
     */
    println(for {
      p1 <- Pole(0, 0).landLeft(1)
      p2 <- p1.landRight(4)
      p3 <- p2.landLeft(-1) // fails here
      p4 <- p3.landRight(-2)
      p5 <- p4.landRight(-2)
    } yield p5)

    // None
    println(for {
      start <- Monad[Option].point(Pole(0, 0))
      first <- start.landLeft(2)
      _ <- (none: Option[Pole])
      second <- first.landRight(2)
      third <- second.landLeft(1)
    } yield third)

    // Instead of making functions that ignore their input and just return a predetermined monadic value,
    // we can use the >> function.
    /*
    scala> (none: Option[Int]) >> 3.some // restart with none
    res25: Option[Int] = None

    scala> 3.some >> 4.some // ignore 3
    res26: Option[Int] = Some(4)

    scala> 3.some >> (none: Option[Int]) // ignore 3
    res27: Option[Int] = None
     */
    // plantage de : cf. http://eed3si9n.com/learning-scalaz/Walk+the+line.html
    // Monad[Option].point(Pole(0, 0)) >>= {_.landLeft(1)} >> (none: Option[Pole]) >>= {_.landRight(1)}
    // car the operator >> precedence.
    println(Monad[Option].point(Pole(0, 0)).>>=({_.landLeft(1)}).>>(none: Option[Pole]).>>=({_.landRight(1)}))
    // Or recognize the precedence issue and place parens around just the right place:
    println((Monad[Option].point(Pole(0, 0)) >>= {_.landLeft(1)}) >> (none: Option[Pole]) >>= {_.landRight(1)})


  }

  /*
  Whereas the Maybe monad is for values with an added context of failure, and the list monad is for nondeterministic
  values, Writer monad is for values that have another value attached that acts as a sort of log value.
  cf. http://eed3si9n.com/learning-scalaz/Writer.html

  sealed trait WriterT[F[+_], +W, +A] { self =>
    val run: F[(W, A)]

    def written(implicit F: Functor[F]): F[W] =
      F.map(run)(_._1)
    def value(implicit F: Functor[F]): F[A] =
      F.map(run)(_._2)
  }

  trait WriterTFunctions {
    def writerT[F[+_], W, A](v: F[(W, A)]): WriterT[F, W, A] = new WriterT[F, W, A] {
      val run = v
    }

    import StoreT._

    def writer[W, A](v: (W, A)): Writer[W, A] =
      writerT[Id, W, A](v)

    def tell[W](w: W): Writer[W, Unit] = writer(w -> ())

    def put[F[+_], W, A](value: F[A])(w: W)(implicit F: Functor[F]): WriterT[F, W, A] =
      WriterT(F.map(value)(a => (w, a)))

    /** Puts the written value that is produced by applying the given function into a writer transformer and associates with `value` */
    def putWith[F[+_], W, A](value: F[A])(w: A => W)(implicit F: Functor[F]): WriterT[F, W, A] =
      WriterT(F.map(value)(a => (w(a), a)))

  }

  trait WriterOps[A] extends Ops[A] {
    def set[W](w: W): Writer[W, A] = WriterT.writer(w -> self)

    def tell: Writer[A, Unit] = WriterT.tell(self)
  }

  scalaz/package.scala :
  ------------------------------
  type Writer[+W, +A] = WriterT[Id, W, A]
  type Unwriter[+W, +A] = UnwriterT[Id, W, A]

   */
  def testWriterMonad() {
    import syntax.writer._
    import syntax.show._
    import std.vector._
    import std.anyVal._ // default shows

    println(3.set("Smallish gang.")) //scalaz.WriterTFunctions$$anon$23@53e082bb
    println(3.set("Smallish gang.").run)// (Smallish gang.,3)
    println("something".tell) // scalaz.WriterTFunctions$$anon$23@38a6ee02
    println("something".tell.run) // (String, Unit) = (something,())


    def gcd(a: Int, b: Int): Writer[Vector[String], Int] =
      if (b == 0) for {
        _ <- Vector("Finished with " + a.shows).tell
      } yield a
      else for {
        _ <- Vector(a.shows + " mod " + b.shows + " = " + (a % b).shows).tell
        result <- gcd(b, a % b)
      } yield result

    println(gcd(8, 3).run) // (Vector(8 mod 3 = 2, 3 mod 2 = 1, 2 mod 1 = 0, Finished with 1),1)


    def testA: Writer[Vector[String], Int] = for {
      _ <- Vector("a").tell
      _ <- Vector("b").tell
      _ <- Vector("c").tell
    } yield 1000

    println(testA.run) // (Vector(a, b, c),1000)

    def invokeService(a: Int): Writer[Vector[String], Int] = for {
      _ <- Vector("Début invokeService").tell
    } yield 1000 + a

    def testB: Writer[Vector[String], Int] = for {
      _ <- Vector("Début testB").tell
      _ <- Vector("b").tell
      b <- invokeService(4)
      _ <- Vector("c").tell
      _ <- Vector("Fin").tell
    } yield b

    println(testB.run) // (Vector(a, b, c),1000)

  }

  /*
  import scalaz._, Scalaz._

  object RWSExample extends App {
    case class Config(port: Int)

    def log[R, S](msg: String): ReaderWriterState[R, List[String], S, Unit] =
      ReaderWriterStateT {
        case (r, s) => (msg.format(r, s) :: Nil, (), s).point[Identity]
      }

    def invokeService: ReaderWriterState[Config, List[String], Int, Int] =
      ReaderWriterStateT {
        case (cfg, invocationCount) => (
          List("Invoking service with port " + cfg.port),
          scala.util.Random.nextInt(100),
          invocationCount + 1
        ).point[Identity]
      }

    val program: RWS[Config, List[String], Int, Int] = for {
      _   <- log("Start - r: %s, s: %s")
      res <- invokeService
      _   <- log("Between - r: %s, s: %s")
      _   <- invokeService
      _   <- log("Done - r: %s, s: %s")
    } yield res

    val Need(logMessages, result, invocationCount) = program run (Config(443), 0)
    println("Result: " + result)
    println("Service invocations: " + invocationCount)
    println("Log: %n%s".format(logMessages.mkString("\t", "%n\t".format(), "")))
  }
   */

}
