#include "stm32f0xx.h"

int main(void)
{
  RCC->AHBENR |= RCC_AHBENR_GPIOCEN;
  GPIOC->MODER |= (1 << (13*2));

  while(1)
  {
    GPIOC->ODR ^= (1 << 13);
    for(volatile int i = 0; i < 500000; i++);
  }
}