from Feature_01 import return_even
from Feature_02 import return_odd


def build_lists(values):
    return return_even(values), return_odd(values)


if __name__ == "__main__":
    import streamlit as st

    original_list = list(range(10))
    even_list, odd_list = build_lists(original_list)

    st.write("Hello world")
    st.write("Hello world again")
    st.write(even_list)
    st.write(odd_list)

    print("Hello Guys")


