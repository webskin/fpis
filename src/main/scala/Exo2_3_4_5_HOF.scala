package fpis


/**
 * Created by Mickaël Gauvin on 2/10/14.
 */
object Exo2_3_4_5_HOF {

  def curry[A,B,C](f: (A, B) => C): A => (B => C) = a => b => f(a, b)

  def uncurry[A,B,C](f: A => B => C): (A, B) => C = (a, b) => f(a)(b)

  /**
   * This is such a common thing to want to do that Scala's standard library provides
   * compose as a method on Function1 (the interface for functions that take one
   * argument). To compose two functions f and g , you simply say f compose g 9 .
   * It also provides an andThen method. f andThen g is the same as g
   * compose f
   */
  def compose[A,B,C](f: B => C, g: A => B): A => C = a => f(g(a))

}
