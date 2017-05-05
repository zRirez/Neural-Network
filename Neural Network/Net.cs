using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;

namespace Neural_Network
{
    class Net
    {
        double[,] wih;
        double[,] who;

        double[] hidden;
        double[] answer;
        double[] target;

        double[] errh;
        double[] erro;

        uint numInputNeurons;
        uint numHiddenNeurons;
        uint numOutputNeurons;

        double stepLearning = 0.001f;

        public Net(uint inputNeurons, uint hiddenNeurons, uint outputNeurons)
        {
            numInputNeurons = inputNeurons;
            numHiddenNeurons = hiddenNeurons;
            numOutputNeurons = outputNeurons;

            wih = new double[inputNeurons, hiddenNeurons];
            who = new double[hiddenNeurons, outputNeurons];

            hidden = new double[numHiddenNeurons];
            answer = new double[numOutputNeurons];
            target = new double[numOutputNeurons];

            errh = new double[hiddenNeurons];
            erro = new double[outputNeurons];

            FileInfo fwih = new FileInfo(".Weight Input-Hidden.txt");
            if (!fwih.Exists)
            {
                Console.WriteLine("Write");
                RandomizeWeight(ref wih, inputNeurons, hiddenNeurons);
                SaveWeightToFile(ref wih, inputNeurons, hiddenNeurons, fwih.Name);
                PrintArr2D(ref wih, inputNeurons, hiddenNeurons);
            }
            else
            {
                Console.WriteLine("Read");
                UploadWeightFromFile(ref wih, inputNeurons, hiddenNeurons, fwih.Name);
                PrintArr2D(ref wih, inputNeurons, hiddenNeurons);
            }

            FileInfo fwho = new FileInfo(".Weight Hidden-Output.txt");
            if (!fwho.Exists)
            {
                Console.WriteLine("Write");
                RandomizeWeight(ref who, hiddenNeurons, outputNeurons);
                SaveWeightToFile(ref who, hiddenNeurons, outputNeurons, fwho.Name);
                PrintArr2D(ref who, hiddenNeurons, outputNeurons);
            }
            else
            {
                Console.WriteLine("Read");
                UploadWeightFromFile(ref who, hiddenNeurons, outputNeurons, fwho.Name);
                PrintArr2D(ref who, hiddenNeurons, outputNeurons);
            }
        }



        void RandomizeWeight(ref double[,] arr, uint imax, uint jmax)
        {
            Thread.Sleep(1);
            Random rn = new Random();

            for (uint i = 0; i < imax; ++i)
                for (uint j = 0; j < jmax; ++j)
                    arr[i, j] = (rn.Next(0, 1001) / 1000.0) - 0.5;
        }



        void SaveWeightToFile(ref double[,] arr, uint imax, uint jmax, string name)
        {
            using (StreamWriter sw = new StreamWriter(name, false)) // true - добавляет в конец: false - стирает всё до открытия и добавляет в конец до закрытия потока
            {
                for (uint i = 0; i < imax; ++i)
                    for (uint j = 0; j < jmax; ++j)
                        sw.WriteLine("{0:N3}", arr[i, j]);
            }
        }



        void UploadWeightFromFile(ref double[,] arr, uint imax, uint jmax, string name)
        {
            using (StreamReader sr = new StreamReader(name))
            {
                for (uint i = 0; i < imax; ++i)
                    for (uint j = 0; j < jmax; ++j)
                        arr[i, j] = Convert.ToDouble(sr.ReadLine());
            }
        }



        double sigmoid(double val)
        {
            return (1.0 / (1.0 + Math.Exp(-val)));
        }



        // Вычислить и вернуть производную сигмоида для аргумента значения.
        double sigmoidDerivative(double val)
        {
            return (val * (1.0 - val));
        }



        void PrintArr1D(ref double[] arr, uint imax)
        {
            for (uint i = 0; i < imax; ++i)
                Console.Write("{0, 8:N3}", arr[i]);
            Console.WriteLine();
        }



        void PrintArr1D(ref uint[] arr, uint imax)
        {
            for (uint i = 0; i < imax; ++i)
                Console.Write("{0, 8:N3}", arr[i]);
            Console.WriteLine();
        }



        void PrintArr2D(ref double[,] arr, uint imax, uint jmax)
        {
            for (uint i = 0; i < imax; ++i)
            {
                for (uint j = 0; j < jmax; ++j)
                    Console.Write("{0, 8:N3}", arr[i, j]);
                Console.WriteLine();
            }
        }



        public void Run(uint[] inputs)
        {
            /////
            //Console.WriteLine("\nInput:");
            //PrintArr1D(ref inputs, numInputNeurons);
            /////

            /* Calculate input to hidden layer */
            for (uint hid = 0; hid < numHiddenNeurons; ++hid)
            {
                double sum = 0.0;
                for (uint inp = 0; inp < numInputNeurons; ++inp)
                {
                    sum += inputs[inp] * wih[inp, hid];
                }
                hidden[hid] = sigmoid(sum);
            }

            /////
            //Console.WriteLine("hidden:");
            //PrintArr1D(ref hidden, numHiddenNeurons);
            /////


            /* Calculate the hidden to output layer */
            for (uint outp = 0; outp < numOutputNeurons; ++outp)
            {
                double sum = 0.0;
                for (uint hid = 0; hid < numHiddenNeurons; ++hid)
                {
                    sum += hidden[hid] * who[hid, outp];
                }
                answer[outp] = sigmoid(sum);
            }

            /////
            //Console.WriteLine("answer:");
            //PrintArr1D(ref answer, numInputNeurons);
            /////

            Action();
        }



        public void Teach()
        {
            uint[] inputs = new uint[numInputNeurons];
            uint ans;

            uint maxEpoch = 10000; //40000
            uint maxSet = 72;

            FileInfo fSamples = new FileInfo(".Examples of training.txt");
            if (!fSamples.Exists)
            {
                Console.WriteLine("\nFile with examples not faund.");
                using (StreamWriter sw = new StreamWriter(fSamples.Name, false)) { }
                Console.WriteLine("It was created, but it's empty.");
                Console.WriteLine("Teaching didn't happen.");
                return;
            }
            else
            {
                for (uint epoch = 0; epoch < maxEpoch; ++epoch)
                {
                    using (StreamReader sr = new StreamReader(fSamples.Name))
                    {
                        for (uint set = 0; set < maxSet; ++set)
                        {
                            // Считываем с файла примеры и делим строку на переменные (входа\выхода)
                            string buf = Convert.ToString(sr.ReadLine());
                            string[] bufs = buf.Split(Convert.ToChar(9));

                            // Конвертируем и получаем готовые результаты (входа\выхода)
                            for (uint i = 0; i < numInputNeurons; ++i)
                                inputs[i] = Convert.ToUInt32(bufs[i]);

                            ans = Convert.ToUInt32(bufs[numInputNeurons]);
                            Array.Clear(target, 0, (int)numOutputNeurons);
                            target[ans] = 1;

                            Run(inputs);

                            double err = 0.0;
                            for (uint i = 0; i < numOutputNeurons; ++i)
                            {
                                err += Math.Pow((target[i] - answer[i]), 2);
                            }
                            err = err / numOutputNeurons;

                            if ((epoch % 1000 == 0) && (set == 0))
                                Console.WriteLine("{0,3}%  Err = {1:N5}", epoch * 100 / maxEpoch, err);

                            RecalculationErrorsAndWeights(inputs);
                        }
                    }
                }
                //Console.WriteLine("\nNew weights: ");
                //PrintArr2D(ref wih, numInputNeurons, numHiddenNeurons);
                //Console.WriteLine();
                //PrintArr2D(ref who, numHiddenNeurons, numOutputNeurons);
                Console.WriteLine("Finish of studies.");
            }
        }



        void RecalculationErrorsAndWeights(uint[] inputs)
        {
            //Вычислить ошибку выходного уровня
            for (uint outp = 0; outp < numOutputNeurons; ++outp)
            {
                erro[outp] = (target[outp] - answer[outp]) * sigmoidDerivative(answer[outp]);
            }

            //Вычислите скрытую ошибку слоя
            for (uint hid = 0; hid < numHiddenNeurons; ++hid)
            {
                errh[hid] = 0.0;
                for (uint outp = 0; outp < numOutputNeurons; ++outp)
                {
                    errh[hid] += erro[outp] * who[hid, outp];
                }
                errh[hid] *= sigmoidDerivative(hidden[hid]);
            }

            //Обновление весов для выходного слоя
            for (uint outp = 0; outp < numOutputNeurons; ++outp)
            {
                for (uint hid = 0; hid < numHiddenNeurons; ++hid)
                {
                    who[hid, outp] += stepLearning * erro[outp] * hidden[hid];
                }
            }

            //Обновите весы скрытого слоя
            for (uint hid = 0; hid < numHiddenNeurons; ++hid)
            {
                for (uint inp = 0; inp < numInputNeurons; ++inp)
                {
                    wih[inp, hid] += stepLearning * errh[hid] * inputs[inp];
                }
            }
        }



        public void Testing()
        {
            uint[] inputs = new uint[numInputNeurons];
            uint ans;

            uint count = 0;
            uint numTest = 10;

            FileInfo fSamples = new FileInfo(".Examples of testing.txt");
            if (!fSamples.Exists)
            {
                Console.WriteLine("\nFile with examples not faund.");
                using (StreamWriter sw = new StreamWriter(fSamples.Name, false)) { }
                Console.WriteLine("It was created, but it's empty.");
                Console.WriteLine("Testing didn't happen.");
                return;
            }
            else
            {
                using (StreamReader sr = new StreamReader(fSamples.Name))
                {

                    for (uint iterTest = 0; iterTest < numTest; ++iterTest)
                    {
                        // Считываем с файла примеры и делим строку на переменные (входа\выхода)
                        string buf = Convert.ToString(sr.ReadLine());
                        string[] bufs = buf.Split(Convert.ToChar(9));

                        // Конвертируем и получаем готовые результаты (входа\выхода)
                        for (uint i = 0; i < numInputNeurons; ++i)
                            inputs[i] = Convert.ToUInt32(bufs[i]);

                        ans = Convert.ToUInt32(bufs[numInputNeurons]);
                        Array.Clear(target, 0, (int)numOutputNeurons);
                        target[ans] = 1;

                        Run(inputs);

                        double err = 0.0;
                        for (uint i = 0; i < numOutputNeurons; ++i)
                        {
                            err += Math.Pow((target[i] - answer[i]), 2);
                        }
                        err = err / numOutputNeurons;

                        if (Translater(ans) == Action())
                        {
                            Console.WriteLine("\n {0}) True", iterTest + 1);
                            ++count;
                        }
                        else Console.WriteLine("\n {0}) False", iterTest + 1);
                        Console.WriteLine("Right answer: '{0}'\nTesting answer: '{1}'", Translater(ans), Action());

                    }
                }
            }
            Console.WriteLine("\n   {0}% right", count * 100 / numTest);
        }



        public string Action()
        {
            uint iter = 0;
            double max = answer[0];

            for (uint i = 1; i < numOutputNeurons; ++i)
            {
                if (max < answer[i])
                {
                    max = answer[i];
                    iter = i;
                }
            }
            return Translater(iter);
        }



        string Translater(uint i)
        {
            switch (i)
            {
                case 0:
                    return "Attack (Атаковать)";
                case 1:
                    return "Hiding in ambush (Прятаться в засаде)";
                case 2:
                    return "Retreat (Отступать)";
                case 3:
                    return "Search for supplies (Искать припасы)";
                default:
                    return "Not definitely (Не определенно)";
            }
        }


    }
}