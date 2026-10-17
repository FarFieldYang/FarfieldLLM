#pragma once

#include <memory>

class ADimpl{

};

class ADvalue{
private:
    std::shared_ptr<ADimpl> impl_;

public:

};