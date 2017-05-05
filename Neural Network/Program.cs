using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Neural_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            uint inputNeurons = 4;
            uint hiddenNeurons = 3;
            uint outputNeurons = 4;

            Net net = new Net(inputNeurons, hiddenNeurons, outputNeurons);

            while (true)
            {
                Console.Write("\nMenu: ");
                string key = Console.ReadLine();

                if (key == "1")
                {
                    Console.WriteLine("\nEnter the incoming signals:");
                    net.Run(InputtingSignals(inputNeurons));
                    Console.WriteLine(net.Action());
                }

                if (key == "2")
                {
                    Console.WriteLine("\nTeaching the neural network..");
                    net.Teach();
                }

                if (key == "3")
                {
                    Console.WriteLine("\nTesting the neural network..");
                    net.Testing();
                }



            }
        }



        static uint[] InputtingSignals(uint inputNeurons)
        {
            uint[] inputSignals = new uint[inputNeurons];
            for (uint i = 0; i < inputNeurons; ++i)
            {
                switch (i)
                {
                    case 0:
                        Console.Write("Health: ");
                        break;
                    case 1:
                        Console.Write("Gun: ");
                        break;
                    case 2:
                        Console.Write("Cartridge: ");
                        break;
                    case 3:
                        Console.Write("Enemies: ");
                        break;
                }
                uint res;
                while (uint.TryParse(Console.ReadLine(), out res))
                {
                    inputSignals[i] = res;
                    break;
                }
            }
            return inputSignals;
        }



    }
}
