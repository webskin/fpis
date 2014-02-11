package fpinscala.datastructures

sealed trait List[+A] // `List` data type, parameterized on a type, `A`
case object Nil extends List[Nothing] // A `List` data constructor representing the empty list
case class Cons[+A](head: A, tail: List[A]) extends List[A] // Another data constructor, representing nonempty lists. Note that `tail` is another `List[A]`, which may be `Nil` or another `Cons`.

object List { // `List` companion object. Contains functions for creating and working with lists.


  val x = List(1,2,3,4,5) match {
    case Cons(x, Cons(2, Cons(4, _))) => x
    case Nil => 42
    case Cons(x, Cons(y, Cons(3, Cons(4, _)))) => x + y // => 3
    case Cons(h, t) => h + List.sum(t) // => 15
    case _ => 101
  }

  val y = List(1,2,3,4,5) match {
    case Cons(x, Cons(2, Cons(4, _))) => x
    case Nil => 42
    case Cons(h, t) => h + List.sum(t) // => 15
    case Cons(x, Cons(y, Cons(3, Cons(4, _)))) => x + y // => 3 unreachable code
    case _ => 101
  }

  def sum(ints: List[Int]): Int = ints match { // A function that uses pattern matching to add up a list of integers
    case Nil => 0 // The sum of the empty list is 0.
    case Cons(x,xs) => x + sum(xs) // The sum of a list starting with `x` is `x` plus the sum of the rest of the list.
  }

  def product(ds: List[Double]): Double = ds match {
    case Nil => 1.0
    case Cons(0.0, _) => 0.0
    case Cons(x,xs) => x * product(xs)
  }

  def apply[A](as: A*): List[A] = // Variadic function syntax
    if (as.isEmpty) Nil
    else Cons(as.head, apply(as.tail: _*))

  def append[A](a1: List[A], a2: List[A]): List[A] =
    a1 match {
      case Nil => a2
      case Cons(h,t) => Cons(h, append(t, a2))
    }

  def foldRight[A,B](l: List[A], z: B)(f: (A, B) => B): B = // Utility functions
    l match {
      case Nil => z
      case Cons(x, xs) => f(x, foldRight(xs, z)(f))
    }

  def sum2(l: List[Int]) =
    foldRight(l, 0)((x,y) => x + y)

  def product2(l: List[Double]) =
    foldRight(l, 1.0)(_ * _) // `_ * _` is more concise notation for `(x,y) => x * y`, see sidebar


/*

  def tail[A](l: List[A]): List[A] = l match {
    case Nil => Nil
    case Cons(x, xs) => xs
  }
*/


  /*
  Although we could return `Nil` when the input list is empty,
  we choose to throw an exception instead. This is a somewhat subjective choice.
  In our experience taking the tail of an empty list is often a bug, and silently
  returning a value just means this bug will be discovered later, further from the place where it was introduced.
  It's generally good practice when pattern matching to use '_' for any variables you don't
  intend to use on the right hand side of a pattern. It makes it clear the value isn't relevant.
  */
  def tail[A](l: List[A]): List[A] =
    l match {
      case Nil => sys.error("tail of empty list")
      case Cons(_, t) => t
    }

  /*
  def setHead[A](l: List[A])(h: A): List[A] = Cons(h, tail(l))
  */

  /*
  If a function body consists solely of a match expression,
  we'll often put the match on the same line as the function signature,
  rather than introducing another level of nesting.
  */
  def setHead[A](l: List[A])(h: A): List[A] = l match {
    case Nil => sys.error("setHead on empty list")
    case Cons(_,t) => Cons(h,t)
  }

  /*
  def drop[A](l: List[A], n: Int): List[A] = {
    if (n == 0) {
      l
    } else {
      drop(tail(l), n - 1)
    }
  }
  */

  /*
  Again, it is somewhat subjective whether to throw an exception when asked to drop more elements than the list contains. The usual default for `drop` is not to throw an exception, since it is typically used in cases where this is not indicative of a programming error. If you pay attention to how you use `drop`, it is often in cases where the length of the input list is unknown, and the number of elements to be dropped is being computed from something else. If `drop` threw an exception, we'd have to first compute or check the length and only drop up to that many elements.
  */
  def drop[A](l: List[A], n: Int): List[A] =
    if (n <= 0) l
    else l match {
      case Nil => Nil
      case Cons(_,t) => drop(t, n-1)
    }

  /*
  def dropWhile[A](l: List[A], f: A => Boolean): List[A] = l match {
    case Nil => Nil
    case Cons(h, t) if f(h) => dropWhile(t, f)
    case _ => l
  }
  */

  /*
  Somewhat overkill, but to illustrate the feature we are using a _pattern guard_,
  to only match a `Cons` whose head satisfies our predicate, `f`.
  The syntax is simply to add `if <cond>` after the pattern, before the `=>`,
  where `<cond>` can use any of the variables introduced by the pattern.
  */
  def dropWhile[A](l: List[A], f: A => Boolean): List[A] =
    l match {
      case Cons(h,t) if f(h) => dropWhile(t, f)
      case _ => l
    }

  /*
  It's a little noisy that we need to state the type of x is Int here. The first
argument to dropWhile is a List[Int] which forces the second argument to
accept an Int . Scala can infer this fact if we group dropWhile into two
argument lists:
   */
  def dropWhile2[A](l: List[A])(f: A => Boolean): List[A] =
    l match {
      case Cons(h,t) if f(h) => dropWhile2(t)(f)
      case _ => l
    }

  /*
  Notice we are copying the entire list up until the last element. Besides being inefficient, the natural recursive solution will use a stack frame for each element of the list, which can lead to stack overflows for large lists (can you see why?). With lists, it's common to use a temporary, mutable buffer internal to the function (with lazy lists or streams which we discuss in chapter 5, we don't normally do this). So long as the buffer is allocated internal to the function, the mutation is not observable and RT is preserved.

  Another common convention is to accumulate the output list in reverse order, then reverse it at the end, which does not require even local mutation. We will write a reverse function later in this chapter.
  */
  def init[A](l: List[A]): List[A] =
    l match {
      case Nil => sys.error("init of empty list")
      case Cons(_,Nil) => Nil
      case Cons(h,t) => Cons(h,init(t))
    }
  def init2[A](l: List[A]): List[A] = {
    import collection.mutable.ListBuffer
    val buf = new ListBuffer[A]
    @annotation.tailrec
    def go(cur: List[A]): List[A] = cur match {
      case Nil => sys.error("init of empty list")
      case Cons(_,Nil) => List(buf.toList: _*)
      case Cons(h,t) => buf += h; go(t)
    }
    go(l)
  }

  def length[A](l: List[A]): Int = sys.error("todo")

  def foldLeft[A,B](l: List[A], z: B)(f: (B, A) => B): B = sys.error("todo")

  def map[A,B](l: List[A])(f: A => B): List[B] = sys.error("todo")
}

