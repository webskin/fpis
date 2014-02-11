package fpis

import org.specs2.mutable._

import fpinscala.datastructures._

/**
 * Created by Mickaël Gauvin on 2/11/14.
 */
class Exo3_1_Spec extends Specification {

  "x" should {
    "be equal to 3" in {
      List.x must be equalTo(3)
    }
  }

  "y" should {
    "be equal to 15" in {
      List.y must be equalTo(15)
    }
  }

  "tail(List(1,2,3,4,5))" should {
    "be equal to List(2,3,4,5)" in {
      List.tail(List(1,2,3,4,5)) must be equalTo(List(2,3,4,5))
    }
  }

  "setHead(10, List(1,2,3,4,5))" should {
    "be equal to List(10, 2,3,4,5)" in {
      List.setHead(List(1,2,3,4,5))(10) must be equalTo(List(10,2,3,4,5))
    }
  }


}
