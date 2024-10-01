#pragma once
#include <bit>
#include <iostream>
#include <concepts>

char32_t ParseChar(const char8_t* utf8, std::uint8_t &size)
{
    size = std::countl_one(static_cast<std::uint8_t>(*utf8));
    switch (size)
    {
    case 0: 
        size = 1;
        return *utf8;
    case 2: return static_cast<std::uint32_t>(utf8[0] & 0b00011111) << 6 | static_cast<std::uint32_t>(utf8[1] & 0b00111111);
    case 3: return static_cast<std::uint32_t>(utf8[0] & 0b00001111) << 12 | static_cast<std::uint32_t>(utf8[1] & 0b00111111) << 6 | static_cast<std::uint32_t>(utf8[2] & 0b00111111);
    case 4: return static_cast<std::uint32_t>(utf8[0] & 0b00000111) << 18 | static_cast<std::uint32_t>(utf8[1] & 0b00111111) << 12 | static_cast<std::uint32_t>(utf8[2] & 0b00111111) << 6 | static_cast<std::uint32_t>(utf8[3] & 0b00111111);
    default: return 0;
    }
}

char32_t ParseChar(const char16_t* utf16, std::uint8_t& size)
{

}

void EncodeChar(char32_t character, char8_t* dest, std::uint8_t &size)
{
    if (character <= 0x7F)
    {
        dest[0] = character & 0b01111111;
        size = 1;
    }
    else if (character <= 0x7FF)
    {
        dest[1] = character & 0b00111111;
        dest[0] = (character << 6) & 0b00011111;
        size = 2;
    }
    else if (character <= 0xFFFF)
    {
        dest[2] = character & 0b00111111;
        dest[1] = (character << 6) & 0b00111111;
        dest[0] = (character << 12) & 0b00001111;
        size = 3;
    }
    else if (character <= 0x10FFFF)
    {
        dest[3] = character & 0b00111111;
        dest[2] = (character << 6) & 0b00111111;
        dest[1] = (character << 12) & 0b00111111;
        dest[0] = (character << 18) & 0b00000111;
        size = 4;
    }
    else
    {
        size = 0;
    }
}

void EncodeChar(char32_t character, char16_t* dest, std::uint8_t& size)
{
    
}