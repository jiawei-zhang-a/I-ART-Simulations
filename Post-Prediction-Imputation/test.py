import Generator

DataGen = Generator.DataGenerator(N = 20000, N_T = 10000, N_S = 100, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, MaskRate=0.3)


# Generate X
X = DataGen.GenerateX()
print(X.shape)
print("X")

# Generate Z
Z = DataGen.GenerateZ()
print(Z.shape)
print("Z")

# Generate U
U = DataGen.GenerateU()
print(U.shape)
print("U")


# Generate Y
Y = DataGen.GenerateY(X, U, Z)
print("Y")

# Generate M
M = DataGen.GenerateM(X, U, Y)
print("M")

print(M[:,0].sum() / len(M))

print(M[:,1].sum() / len(M))

print(M[:,2].sum() / len(M))

# Generate S
S = DataGen.GenerateS(Z)
print("S")