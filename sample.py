class Person:
    count = 0
    def __init__ (self, name):
        self.name = name
        Person.count += 1
    @classmethod
    def get_count(cls):
        return cls.count
        

a = Person("小米")
b = Person("大米")
#print (Person.get_count())  
#print (Person.count)      
res = list(map(int, input().split('-')))
print(res)