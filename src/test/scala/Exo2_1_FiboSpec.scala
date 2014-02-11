package fpis

import org.specs2.mutable._

import Exo2_1_Fibo._

/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
class Exo2_1_FiboSpec extends Specification {

  "fib(0)" should {
    "be equal to 0" in {
      fib(0) must be equalTo(0)
    }
  }

  "fib(1)" should {
    "be equal to 1" in {
      fib(1) must be equalTo(1)
    }
  }

  "fib(2)" should {
    "be equal to 1" in {
      fib(2) must be equalTo(1)
    }
  }

  "fib(3)" should {
    "be equal to 2" in {
      fib(3) must be equalTo(2)
    }
  }

  "fib(6)" should {
    "be equal to 8" in {
      fib(6) must be equalTo(8)
    }
  }
}
