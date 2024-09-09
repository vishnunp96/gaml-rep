from lark import Lark

parser = Lark('''

// Final measurement production
measurement : number units? junk?

// Rule to combine units
units : (unit|inverseunit)+

// Individual unit rules
unit : UNIT
inverseunit : ("/"|"per") UNIT

// Unit productions (simplified)
UNIT.2 : (SIPREFIX? SIUNIT) | /\%/
SIUNIT.2 : "m" | "pc" | "s"
SIPREFIX.2 : "k" | "M"

number : NUMBER
NUMBER.2 : /[\+\-]?[0-9\.]+/

junk : OTHER+
OTHER : /[^\s]+/

%import common.WS
%ignore WS

''', parser="lalr", lexer="contextual", start='measurement')

print(parser.parse('45 km / s / Mpc').pretty())
print(parser.parse('45 km/s/Mpc').pretty())

print(parser.parse('45 is just a number').pretty())
print(parser.parse('45 km / s / Mpc and then more text .').pretty())
