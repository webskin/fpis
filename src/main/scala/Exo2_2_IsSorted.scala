package fpis

/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
object Exo2_2_IsSorted {

  def isSorted[A](as: Array[A], gt: (A,A) => Boolean): Boolean = {

    @annotation.tailrec
    def loop(pos: Int): Boolean = {
      if (pos >= as.length - 1) true
      else if (gt(as(pos), as(pos + 1))) loop(pos + 1)
      else false
    }

    loop(0)
  }

  /*// Solution
  // Exercise 2: Implement a polymorphic function to check whether
  // an `Array[A]` is sorted
  def isSorted[A](as: Array[A], gt: (A,A) => Boolean): Boolean = {
    @annotation.tailrec
    def go(i: Int, prev: A): Boolean =
      if (i == as.length) true
      else if (gt(as(i), prev)) go(i + 1, as(i))
      else false
    if (as.length == 0) true
    else go(1, as(0))
  }*/

}
