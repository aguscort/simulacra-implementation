class Habitante:
    def __init__(self, nombre, descripcion):
        self.nombre = nombre
        self.descripcion = descripcion

    def __str__(self):
        return f"Habitante: {self.nombre}, {self.descripcion}"
class Elemento:
    def __init__(self, nombre, descripcion, objetos=None):
        self.nombre = nombre
        self.descripcion = descripcion
        self.objetos = objetos if objetos is not None else []

    def agregar_objeto(self, objeto):
        self.objetos.append(objeto)

    def __str__(self):
        objetos_str = '\n\t\t'.join(str(objeto) for objeto in self.objetos)
        return f"Elemento: {self.nombre} - {self.descripcion}\n\t\tObjetos en {self.nombre}:\n\t\t{objetos_str}"

class Habitacion:
    def __init__(self, nombre, elementos=None, habitantes=None):
        self.nombre = nombre
        self.elementos = elementos if elementos is not None else []
        self.habitantes = habitantes if habitantes is not None else []

    def agregar_elemento(self, elemento):
        self.elementos.append(elemento)

    def agregar_habitante(self, habitante):
        self.habitantes.append(habitante)

    def __str__(self):
        elementos_str = '\n\t'.join(str(elem) for elem in self.elementos)
        habitantes_str = '\n\t'.join(str(hab) for hab in self.habitantes)
        return f"Habitación: {self.nombre}\n\tElementos:\n\t{elementos_str}\n\tHabitantes:\n\t{habitantes_str}"

class Pueblo:
    def __init__(self, nombre, habitaciones=None):
        self.nombre = nombre
        self.habitaciones = habitaciones if habitaciones is not None else []

    def agregar_habitacion(self, habitacion):
        self.habitaciones.append(habitacion)

    def __str__(self):
        habitaciones_str = '\n'.join(str(habitacion) for habitacion in self.habitaciones)
        return f"Pueblo: {self.nombre}\n{habitaciones_str}"


# Crear elementos y habitantes
libro = Elemento("Libro", "Un libro de aventuras.")
laptop = Elemento("Laptop", "Una laptop de alto rendimiento.")
escritorio = Elemento("Escritorio", "Un amplio escritorio de madera.")
escritorio.agregar_objeto(libro)
escritorio.agregar_objeto(laptop)

juan = Habitante("Juan", "Un escritor trabajador.")

# Crear una habitación y agregar elementos y habitantes
oficina = Habitacion("Oficina")
oficina.agregar_elemento(escritorio)
oficina.agregar_habitante(juan)

# Crear el pueblo y agregar habitaciones
pueblo_de_escritores = Pueblo("Pueblo de Escritores")
pueblo_de_escritores.agregar_habitacion(oficina)

# Imprimir la estructura del pueblo
print(pueblo_de_escritores)
