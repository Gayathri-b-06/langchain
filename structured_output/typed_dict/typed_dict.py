from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str
    
p1:Person={'name': 'John', 'age': 30, 'email': 'john@example.com'}

print(p1)

#? typeddict :it is a way of defining a dictionary with specific keys and value types. It allows you to create a structured dictionary that can be used for type checking and code completion in Python.

