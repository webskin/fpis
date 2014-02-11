package fpis

import org.specs2.mutable._

import Exo3_1._

/**
 * Created by Mickaël Gauvin on 2/11/14.
 */
class Exo3_1_Spec extends Specification {

  "x" should {
    "be equal to 3" in {
      x must be equalTo(3)
    }
  }

  "y" should {
    "be equal to 15" in {
      y must be equalTo(15)
    }
  }
}
