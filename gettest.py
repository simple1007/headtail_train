class Test:
    def init(self):
        self.vars = {"abc":34,"def":356}
    
    def __get__(self,obj,owner):
        # print(obj,owner)
        # print(obj.__dict__)
        # obj.__dict__["abc"] = self.vars["abc"]
        return obj.__dict__["abc"]
    def __set__(self,obj,value):
        # print(obj,value)
        # self.vars = value
        for k,v in value.items():
            obj.__dict__[k] = v
        # print(obj.__dict__)
class A:    
    # t = Test()
    # t = Test()
         
    def __init__(self):
        self.vars = {"abc":34,"defg":356}
        
        for k,v in self.vars.items():
            self.__dict__[k] = v       
        
        # self.t.value = self.vars
        # t.vars = self.vars
        # self.t.abc = self.vars
        # self.t.def_ = self.vars["def"]
tt = A()
print(tt.abc)
print(tt.defg)
# print(tt.__dict__)