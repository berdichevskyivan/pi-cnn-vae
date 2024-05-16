import numpy as np

# Digits of Pi (taking the first 256 digits to create a 4x4x4x4 tensor)
pi_digits = [int(digit) for digit in "141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923543434646212536800828110761488643800888315946891727015330748340264604197028630907678960150141288531250439840123385944600"
]

# Ensure we have exactly 256 digits
pi_digits = pi_digits[:256]

# Create a 4x4x4x4 tensor
tensor = np.array(pi_digits).reshape(4, 4, 4, 4)

# Print tensor shape
print("Tensor shape:", tensor.shape)