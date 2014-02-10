/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
object Exo1Fibo {

  def fib(n: Int): Int = {
    def localFib(a: Int, b: Int, remaining: Int): Int = {
      if (remaining == 0) {
        a
      } else {
        localFib(a + b, b, remaining - 1)
      }
    }
    localFib(0, 1, n)
  }
}
