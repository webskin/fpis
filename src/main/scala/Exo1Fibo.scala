package fpis

/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
object Exo1Fibo {

  /*
  def fib(n: Int): Int = {
    def localFib(a: Int, b: Int, remaining: Int): Int = {
      if (remaining == 0) {
        b
      } else {
        localFib(b, a + b, remaining - 1)
      }
    }
    if (n <= 0) {
      0
    } else if (n == 1) {
      1
    } else {
      localFib(1, 1, n - 2)
    }
  }
  */

  // solution
  // 0 and 1 are the first two numbers in the sequence,
  // so we start the accumulators with those.
  // At every iteration, we add the two numbers to get the next one.
  def fib(n: Int): Int = {
    @annotation.tailrec
    def loop(n: Int, prev: Int, cur: Int): Int =
      if (n == 0) prev
      else loop(n - 1, cur, prev + cur)
    loop(n, 0, 1)
  }
}
