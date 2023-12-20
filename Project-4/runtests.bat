main tests/TEST-0.bmp results/TEST-0-Rotate.bmp rotate gaussian 0.785
main tests/TEST-0.bmp results/TEST-0-Translation.bmp translation gaussian -30 0
main tests/TEST-0.bmp results/TEST-0-Shear.bmp shear gaussian 3 1
main tests/TEST-0.bmp results/TEST-0-Mirror.bmp mirror gaussian -1 1
main tests/TEST-0.bmp results/TEST-0-Scale.bmp scale cubic 3 1

main tests/TEST-1.bmp results/TEST-1-Rotate.bmp rotate gaussian 3.1415926
main tests/TEST-1.bmp results/TEST-1-Translation.bmp translation gaussian 0 30
main tests/TEST-1.bmp results/TEST-1-Shear.bmp shear gaussian 1 3
main tests/TEST-1.bmp results/TEST-1-Mirror.bmp mirror gaussian 1 -1
main tests/TEST-1.bmp results/TEST-1-Scale.bmp scale cubic 0.5 1

pause
