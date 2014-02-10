package fpis

import org.specs2.mutable._

import Exo2IsSorted._

/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
class Exo2IsSortedSpec extends Specification {

  "[1, 2, 3, 4, 10]" should {
    "be sorted" in {
      isSorted[Int](Array(1, 2, 3, 4, 10), _ <= _) equals(true)
    }
  }

  "[1, 2, 3, 4, -5, 10]" should {
    "not be sorted" in {
      isSorted[Int](Array(1, 2, 3, 4, -5, 10), _ <= _) equals(false)
    }
  }


}
