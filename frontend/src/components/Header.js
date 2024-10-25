import React, {useEffect} from 'react';
import {Grid, Slide, useScrollTrigger} from "@mui/material";
import Fab from "@mui/material/Fab";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import Typography from "@mui/material/Typography";
import {createPortal} from "react-dom";

function HideOnScroll({ children }) {
  const trigger = useScrollTrigger();

  return (
    <Slide appear={false} direction="down" in={!trigger}>
      {children}
    </Slide>
  );
}

const Header = ({ children }) => {
  return (
    <HideOnScroll>
      <Grid
        left={0}
        top={0}
        width={'100%'}
        sx={{ background: 'var(--bg-color)', position: 'fixed', boxShadow: 'rgba(0, 0, 0, 0.2) 0px 2px 4px -1px, rgba(0, 0, 0, 0.14) 0px 4px 5px 0px, rgba(0, 0, 0, 0.12) 0px 1px 10px 0px' }}
        zIndex={100}
        display={'flex'}
        justifyContent={'center'}
        alignItems={'center'}
        borderRadius={'0 0 var(--border-radius-lg) var(--border-radius-lg)'}
        overflow={'hidden'}
      >
        <Grid
          ml={'auto'}
          mr={'auto'}
          display='flex'
          alignItems='center'
          width={'calc(var(--content-width) - 2 * var(--space-md))'}
          sx={{ background: 'var(--bg-color)' }}
          p={2}
        >
          {children}
        </Grid>
      </Grid>
    </HideOnScroll>
  );
};



export default Header;