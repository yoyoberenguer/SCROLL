
# include <stdio.h>
int main(){

int i=0, ii=0, iii = 0;
int dx = 1;
int dy = -1;
for (i=0; i<192; i++){
  iii = (i + dx * 3) % 8;
  ii = (iii + (8 * 3) * dy) % 8;
  printf("\n%i %i ", iii, ii);
}

}