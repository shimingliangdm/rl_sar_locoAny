#ifndef MATH_STRUCT_HPP
#define MATH_STRUCT_HPP



namespace math_struct
{

template <typename T>
struct Position
{
    T x;
    T y;
    T z;
    Position(T inX, T inY, T inZ)
    {
        x = inX;
        y = inY;
        z = inZ;
    }
};

template <typename T>
struct Quat
{
    T x;
    T y;
    T z;
    T w;
    Quat(T inX, T inY, T inZ, T inW)
    {
        x = inX;
        y = inY;
        z = inZ;
        w = inW;
    }
};

}

#endif