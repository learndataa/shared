#include "stm32f0xx.h"

void UART_Init(void);
void UART_SendChar(char c);
void UART_SendString(char *str);
void delay(uint32_t n);

int main(void)
{
  UART_Init();

  while (1)
  {
    UART_SendString("99\r\n");
    //delay(1000000);
  }
}

void UART_Init(void)
{
  RCC->AHBENR  |= RCC_AHBENR_GPIOAEN;
  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
  GPIOA->MODER  &= ~(3 << (2*2));
  GPIOA->MODER  |=  (2 << (2*2));
  GPIOA->AFR[0] &= ~(0xF << (2*4));
  GPIOA->AFR[0] |=  (1   << (2*4));
  USART2->BRR = 8000000 / 9600;
  USART2->CR1 |= USART_CR1_TE | USART_CR1_UE;
}

void UART_SendChar(char c)
{
  while (!(USART2->ISR & USART_ISR_TXE));
  USART2->TDR = c;
}

void UART_SendString(char *str)
{
  while (*str) UART_SendChar(*str++);
}

void delay(uint32_t n)
{
  for(volatile uint32_t i=0; i<n; i++);
}