


# class EnumBodyDict(dict):
#     """ https://news.ycombinator.com/item?id=5691483 """
#     def __init__(self, *a, **kw):
#         self._keys_accessed = []
#         dict.__init__(self, *a, **kw)
#
#     def __getitem__(self, key):
#         self._keys_accessed.append(key)
#         return dict.__getitem__(self, key)
#
#
# class EnumMeta(type):
#     @classmethod
#     def __prepare__(metacls, name, bases):
#         return EnumBodyDict()
#
#     def __new__(cls, name, bases, classdict):
#         next_enum_value = max(classdict.values()) + 1
#
#         for name in classdict._keys_accessed:
#             if name not in classdict:
#                 classdict[name] = next_enum_value
#                 next_enum_value += 1
#
#         return type.__new__(cls, name, bases, classdict)
#
#
# class Enum(object, metaclass=EnumMeta):
#
#
# # proposed enum implementation here
#
# # class Config(metaclass=Struct):
#
