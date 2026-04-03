using System;
using System.Windows.Forms;
using TorchSharp;
using ChessLLM;

if (torch.cuda.is_available())
    torch.InitializeDeviceType(DeviceType.CUDA);

Application.EnableVisualStyles();
Application.SetCompatibleTextRenderingDefault(false);
Application.Run(new MainForm());
