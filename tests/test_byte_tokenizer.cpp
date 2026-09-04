#include <iostream>
#include <vector>
#include <string>
#include "../tokenizer/byte_tokenizer.h"

int main(){
    std::string text = "hello, FarfieldLLM!  你好，远场大语言模型！😊😊😊";
    ByteTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode(text);
    std::string decode_text = tokenizer.decode(tokens);

    std::cout << text << std::endl;
    for (int token : tokens){
        std::cout << token << ' ';
    }
    std::cout << std::endl;
    std::cout << decode_text << std::endl;
    return 0;
}